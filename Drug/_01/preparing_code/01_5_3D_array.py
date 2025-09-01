import numpy as np
import matplotlib.pyplot as plt
import os

# =============================
# 데이터 로드
# =============================
path = './_data/dacon/drug/data/'
molecule_array = np.load(path + 'molecule_3D(5dimension).npy')

# 첫 번째 분자 선택
single_molecule = molecule_array[0]  # shape: (32, 32, 32, 10)

# =============================
# 저장 폴더
# =============================
save_path = './_data/dacon/drug/image_output_color/'
os.makedirs(save_path, exist_ok=True)

# =============================
# 단면 선택
# =============================
z_index = 16  # 가운데 슬라이스

# 채널 매핑
# Red → C, Green → N, Blue → O
# 나머지 원자는 밝기 강화

slice_r = single_molecule[z_index, :, :, 0]  # 탄소 (C)
slice_g = single_molecule[z_index, :, :, 2]  # 질소 (N)
slice_b = single_molecule[z_index, :, :, 1]  # 산소 (O)

# 나머지 원자들 → 밝기 강화용으로 처리
slice_rest = np.sum(single_molecule[z_index, :, :, 3:], axis=-1) * 0.5

# =============================
# RGB 이미지 만들기
# =============================
rgb_image = np.stack([slice_r, slice_g, slice_b], axis=-1)
rgb_image += slice_rest[..., np.newaxis]  # 밝기 강화

# 값 클리핑
rgb_image = np.clip(rgb_image, 0, 1)

# =============================
# 저장
# =============================
plt.imshow(rgb_image)
plt.title(f'Molecule Z={z_index} RGB')
plt.axis('off')

filename = f'molecule0_z{z_index}_RGB.png'
plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0)
plt.close()

print(f"✅ 컬러 이미지 저장 완료 → {os.path.join(save_path, filename)}")
