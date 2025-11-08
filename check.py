# from data.dataset import AnomalyDataset
# ds_tr = AnomalyDataset(
#     root="/data3/jisu/MFM/datasets/VisA",
#     dataset_name="visa",
#     split="train",
#     image_size=224,
#     return_mask=False,
# )
# print("len(train) =", len(ds_tr))

# # for i in range( len(ds_tr)):
# #     it = ds_tr[i]
# #     if it["label"].item() !=0:
# #         print("[train]", it["label"].item(), it["meta"]["img_path"])
# #         # 출력이 아무것도 안되니, 정상만 잘 들어갔다는 뜻. 

# # for i in range(min(5, len(ds_tr))):
# #     it = ds_tr[i]
# #     print("[train]", it["label"].item(), it["meta"]["img_path"])
# #     # 기대: label=0, 경로에 /Data/Images/Normal/ 포함

# ds_te = AnomalyDataset(
#     root="/data3/jisu/MFM/datasets/VisA",
#     dataset_name="visa",
#     split="test",
#     image_size=224,
#     return_mask=False,
# )
# print("len(test) =", len(ds_te))
# # c0 = sum(int(ds_te[i]["label"].item()==0) for i in range(min(500, len(ds_te))))
# c1 = sum(int(ds_te[i]["label"].item()==1) for i in range(len(ds_te)))
# # print(f"rough test counts (first 500): normal={c0}, anomaly={c1}")
# print("Anomaly 개수:", c1)

# from data.dataset import make_dataloader
# loader = make_dataloader(
# root="/data3/jisu/MFM/datasets/VisA",
# dataset_name="visa",
# split="train",
# batch_size=2,
# num_workers=0,   # 워커 0으로 먼저 확인하면 스택트레이스가 더 깔끔
# image_size=224,
# return_mask=False,
# )
# b = next(iter(loader))
# print(type(b["meta"]), len(b["meta"]))        # list, B
# print(b["image"].shape, b["label"].shape)     # (B,3,H,W) (B,)