import cv2
import numpy as np

# 步驟一：gamma校正
IO = cv2.imread('dde.png')
I = IO.astype(np.float32)/255.0
gamma_values = [1.0, 1.2, 2.0, 4.0, 8.0]
gamma_list = [I]

for gamma in gamma_values[1:]:
    I_corrected = np.power(I ,gamma)
    gamma_list.append(I_corrected)


# 步驟二：計算圖像的亮度(RGB空間中的亮度(強度)，也就是灰階)
L_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in gamma_list]


# 步驟三：生成base層（引導濾波）
radius = 12
epsilon = 0.25
B_list = [cv2.ximgproc.guidedFilter(L, L, radius, epsilon) for L, L in zip(L_list, L_list)]


# 步驟四：生成detail層
D_list = [(I - cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)) for I, B in zip(gamma_list, B_list)]                                                   #手動拓成三維RGB


# 步驟五：計算detail層的權重
detail_weight_list = []

epsilon = 1e-2

for detail_img, L_img in zip(D_list, L_list):
    L_mean = cv2.blur(L_img, (7, 7))
    sigma_D = 0.12
    # 計算detail權重
    detail_weight = np.exp((-((L_mean - 0.5)**2))/(2 * (sigma_D**2)))
    detail_weight = detail_weight
    detail_weight_list.append(detail_weight)

dwl_shape = detail_weight_list[0].shape              #------------------------------------------------------------------------------------------------把串列裡面的五張圖相加
dwl = np.zeros(dwl_shape, dtype=np.float32)
for i in range(len(detail_weight_list)):
    dwl = np.add(dwl, detail_weight_list[i])

detail_weight_list_division = []                     #------------------------------------------------------------------------------------------------創建一個新的串列來存儲結果

for dw in detail_weight_list:                        #------------------------------------------------------------------------------------------------進行除法，將結果加到 detail_weight_list_division 中
    dw_division = dw / dwl
    detail_weight_list_division.append(dw_division)


# 步驟六 : 計算base層的權重
base_weight_list = []

for base_img, L_img in zip(B_list, L_list):
    L_mean = np.mean(L_img)
    sigma_B = 0.5
    # 計算base權重
    base_weight = np.exp((-(((base_img - 0.5)**2) + (6.25*((L_mean - 0.5)**2)))) / (2 * (sigma_B**2)))
    base_weight_list.append(base_weight)

bwl_shape = base_weight_list[0].shape                #------------------------------------------------------------------------------------------------把串列裡面的五張圖相加
bwl = np.zeros(bwl_shape, dtype=np.float32)
for i in range(len(base_weight_list)):
    bwl = np.add(bwl, base_weight_list[i])

base_weight_list_division = []                       #------------------------------------------------------------------------------------------------創建一個新的串列來存儲結果

for bw in base_weight_list:                          #------------------------------------------------------------------------------------------------進行除法，將結果加到 detail_weight_list_division 中
    bw_division = bw / bwl
    base_weight_list_division.append(bw_division)


# 步驟七 : 融合(Fusion) (F)

fusion_image_base = []

for WBk, Bk, in zip(base_weight_list_division, B_list):       #---------------------------------------------------------------------------------------base 融合，存到fusion_image_base裡
    WBk_rgb = cv2.cvtColor(WBk, cv2.COLOR_GRAY2BGR)
    Bk_rgb = cv2.cvtColor(Bk, cv2.COLOR_GRAY2BGR)
    fusion_image = WBk_rgb * Bk_rgb
    fusion_image_base.append(fusion_image)

fib_shape = fusion_image_base[0].shape               #------------------------------------------------------------------------------------------------把串列裡面的五張圖相加
fib = np.zeros(fib_shape, dtype=np.float32)
for i in range(len(fusion_image_base)):
    fib = np.add(fib, fusion_image_base[i])


fusion_image_detail = []

for WDk, Dk in zip(detail_weight_list_division, D_list):       #--------------------------------------------------------------------------------------detail 融合，存到fusion_image_detail裡
    WDk_rgb = cv2.cvtColor(WDk, cv2.COLOR_GRAY2BGR)
    fusion_image = WDk_rgb * Dk
    fusion_image_detail.append(fusion_image)

fid_shape = fusion_image_detail[0].shape               #----------------------------------------------------------------------------------------------把串列裡面的五張圖相加
fid = np.zeros(fid_shape, dtype=np.float32)
for i in range(len(fusion_image_detail)):
    fid = np.add(fid, fusion_image_detail[i])


#融合所有圖像 並截斷
final_fusion_image = fib + (1.1*fid)
final_fusion_image_clip = np.clip(final_fusion_image,0,1)
cv2.imwrite("F.png",(final_fusion_image_clip * 255).astype(np.uint8))
cv2.imshow("F : ",(final_fusion_image_clip * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


# 步驟八 : 飽和度調整 (H)

#融合圖像LS之差 WF
final_fusion_image_clip_255 = (final_fusion_image_clip*255).astype(np.uint8)
HSV_F = cv2.cvtColor(final_fusion_image_clip_255, cv2.COLOR_BGR2HSV)
HSV_F = HSV_F.astype(np.float32)
H_F, S_F, V_F = cv2.split(HSV_F)
S_F = S_F/255
V_F = V_F/255
diff_F = V_F - S_F
WF = 0.0
for i in range(diff_F.shape[0]):
    for j in range(diff_F.shape[1]):
            if diff_F[i, j] > 0.3:
                WF += diff_F[i, j]
print("WF : ",WF)

#原始有霧圖像LS之差 WI
HSV_I = cv2.cvtColor(IO, cv2.COLOR_BGR2HSV)
HSV_I = HSV_I.astype(np.float32)
H_I, S_I, V_I = cv2.split(HSV_I)
S_I = S_I/255
V_I = V_I/255
diff_I = V_I - S_I
WI = 0.0
for i in range(diff_I.shape[0]):
    for j in range(diff_I.shape[1]):
            if diff_I[i, j] > 0.3:
                WI += diff_I[i, j]
print("WI : ",WI)

#飽和度比 tau_s
VVF = np.sum(V_F)
SSF = np.sum(S_F)
VVI = np.sum(V_I)
SSI = np.sum(S_I)
coefficient = WF/WI
tau_s = (VVF - (coefficient*(VVI - SSI)))/SSF
print("tau_s :",tau_s)
if tau_s <= 1:
    output_image = final_fusion_image_clip_255
else:
    HSV_F[:,:,1] *= tau_s
    HSV_F[:,:,1] = np.clip(HSV_F[:, :, 1], 0, 255)
    output_image = HSV_F.astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_HSV2BGR)
cv2.imwrite("H.png",output_image)
cv2.imshow("H : ",output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

