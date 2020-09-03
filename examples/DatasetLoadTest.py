import sys
sys.path.append("../")
from TorchUtils.DatasetGenerator import FromPublicDatasets
from TorchUtils.DatasetGenerator import FromFolder
import cv2


loader = FromFolder.generate_dataloader(r"C:\Users\chino\OneDrive\Desktop\ikura-pattern\pattern", is_single_labels=True, ext="jpg", batch_size=1)


if __name__ == "__main__":
    for images, labels in loader:
        img = images.detach().numpy()[0]
        img = img.transpose((1, 2, 0))
        print(img.shape)
        cv2.imshow("hoge", img)
        while True:
            if cv2.waitKey(20) == ord("q"):
                cv2.destroyAllWindows()
                break
