import torch
import torch.nn.functional as F
import os
import cv2
import glob
import numpy as np
import time
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import convert_state_dict
from data.preprocess.crop_merge_image import stride_integral

os.sys.path.append('./models/UNeXt')
from models.UNeXt.unext import UNext_full_resolution_padding, UNext_full_resolution_padding_L_py_L


def calculate_metrics(gt_path, pred_img):
    """
    Вычисляет SSIM и PSNR между ground truth и предсказанным изображением
    gt_path: путь к ground truth изображению
    pred_img: предсказанное изображение (numpy array, uint8)
    """
    # Загрузка ground truth
    gt = cv2.imread(gt_path)
    if gt is None:
        raise ValueError(f"Не удалось загрузить GT изображение: {gt_path}")

    # Приводим к одному размеру (на всякий случай)
    if gt.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt.shape[1], gt.shape[0]))

    # Конвертируем в grayscale для SSIM (как принято для документов)
    # Если хотите цветной SSIM - используйте multichannel=True
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)

    # SSIM
    ssim_val = ssim(gt_gray, pred_gray, data_range=255)

    # PSNR (можно считать на RGB или на gray)
    psnr_val = psnr(gt, pred_img, data_range=255)

    return ssim_val, psnr_val


def eval_model1_model2_with_metrics(model1, model2, path_list, in_folder, sav_folder, gt_folder):
    """
    Расширенная версия функции инференса с сбором метрик
    """
    ssim_scores = []
    psnr_scores = []
    inference_times = []

    # Создаем папку для сохранения результатов, если нужно
    if not os.path.exists(sav_folder):
        os.mkdir(sav_folder)

    for im_path in tqdm(path_list, desc="Processing images"):
        # Получаем имя файла для поиска GT
        base_name = os.path.basename(im_path)
        gt_path = os.path.join(gt_folder, base_name)

        if not os.path.exists(gt_path):
            print(f"Предупреждение: GT не найден для {base_name}, пропускаем...")
            continue

        # Загрузка и подготовка изображения
        im_org = cv2.imread(im_path)
        if im_org is None:
            print(f"Предупреждение: не удалось загрузить {im_path}, пропускаем...")
            continue

        im_org, padding_h, padding_w = stride_integral(im_org)
        h, w = im_org.shape[:2]
        im = im_org  # используем исходный размер, не ресайзим на 512

        # Инференс с замером времени
        torch.cuda.synchronize()  # Синхронизация для точного измерения
        start_time = time.time()

        with torch.no_grad():
            im_tensor = torch.from_numpy(im.transpose(2, 0, 1) / 255).unsqueeze(0).float().cuda()
            im_org_tensor = torch.from_numpy(im_org.transpose(2, 0, 1) / 255).unsqueeze(0).float().cuda()

            # Model1: предсказание тени
            shadow = model1(im_tensor)
            shadow = F.interpolate(shadow, (h, w))

            # Коррекция освещения
            model1_im = torch.clamp(im_org_tensor / shadow, 0, 1)

            # Model2: финальная реставрация
            pred, _, _, _ = model2(torch.cat((im_org_tensor, model1_im), 1))

            # Постобработка для метрик
            pred_np = pred[0].permute(1, 2, 0).cpu().numpy()
            pred_np = (pred_np * 255).astype(np.uint8)
            pred_np = pred_np[padding_h:, padding_w:]

        torch.cuda.synchronize()  # Синхронизация после инференса
        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        # Сохранение результата (опционально)
        output_path = im_path.replace(in_folder, sav_folder)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, pred_np)

        # Вычисление метрик
        ssim_val, psnr_val = calculate_metrics(gt_path, pred_np)
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)

        # Логирование
        tqdm.write(f"{base_name}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f}dB, Time={inference_time * 1000:.2f}ms")

    return ssim_scores, psnr_scores, inference_times


def main():
    # Пути
    model1_path = 'checkpoints/gcnet/checkpoint.pkl'
    model2_path = 'checkpoints/drnet/checkpoint.pkl'
    img_folder = './distorted/'  # Папка с искаженными изображениями
    gt_folder = './ground_truth/'  # Папка с эталонными изображениями
    sav_folder = './enhanced/'  # Папка для сохранения результатов

    # Проверка существования папок
    for folder, name in [(img_folder, 'distorted'), (gt_folder, 'ground_truth')]:
        if not os.path.exists(folder):
            raise ValueError(f"Папка {name} не найдена: {folder}")

    # Загрузка моделей
    print("Загрузка моделей...")
    model1 = UNext_full_resolution_padding(num_classes=3, input_channels=3, img_size=512).cuda()
    state = convert_state_dict(torch.load(model1_path, map_location='cuda:0')['model_state'])
    model1.load_state_dict(state)

    model2 = UNext_full_resolution_padding_L_py_L(num_classes=3, input_channels=6, img_size=512).cuda()
    state = convert_state_dict(torch.load(model2_path, map_location='cuda:0')['model_state'])
    model2.load_state_dict(state)

    model1.eval()
    model2.eval()

    # Получение списка изображений
    im_paths = glob.glob(os.path.join(img_folder, '*'))
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    im_paths = [p for p in im_paths if p.lower().endswith(supported_ext)]

    print(f"Найдено {len(im_paths)} изображений")

    # Запуск оценки
    ssim_scores, psnr_scores, inference_times = eval_model1_model2_with_metrics(
        model1, model2, im_paths, img_folder, sav_folder, gt_folder
    )

    # Агрегированная статистика
    if len(ssim_scores) > 0:
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ:")
        print("=" * 50)
        print(f"SSIM  - Mean: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
        print(f"PSNR  - Mean: {np.mean(psnr_scores):.2f} dB ± {np.std(psnr_scores):.2f}")
        print(f"Inference Time - Mean: {np.mean(inference_times) * 1000:.2f} ms ± {np.std(inference_times) * 1000:.2f}")
        print(f"Total images processed: {len(ssim_scores)}")

        # Сохранение результатов в файл
        results_file = os.path.join(sav_folder, 'metrics_results.txt')
        with open(results_file, 'w') as f:
            f.write("Image,SSIM,PSNR,InferenceTime_ms\n")
            for i, (img_path, ssim_val, psnr_val, inf_time) in enumerate(
                    zip(im_paths[:len(ssim_scores)], ssim_scores, psnr_scores, inference_times)):
                img_name = os.path.basename(img_path)
                f.write(f"{img_name},{ssim_val:.4f},{psnr_val:.2f},{inf_time * 1000:.2f}\n")
            f.write(f"\nSUMMARY:\n")
            f.write(f"SSIM_mean,{np.mean(ssim_scores):.4f}\n")
            f.write(f"SSIM_std,{np.std(ssim_scores):.4f}\n")
            f.write(f"PSNR_mean,{np.mean(psnr_scores):.2f}\n")
            f.write(f"PSNR_std,{np.std(psnr_scores):.2f}\n")
            f.write(f"Inference_time_mean_ms,{np.mean(inference_times) * 1000:.2f}\n")
            f.write(f"Inference_time_std_ms,{np.std(inference_times) * 1000:.2f}\n")
        print(f"\nРезультаты сохранены в: {results_file}")
    else:
        print("Ошибка: не удалось вычислить метрики ни для одного изображения")


if __name__ == '__main__':
    main()