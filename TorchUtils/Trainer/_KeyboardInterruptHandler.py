import torch
import sys
import colorama

colorama.init()
from colorama import Fore
from ._ModelSaver import save_model


def respond_exeption(model: torch.nn.Module):
    print(Fore.CYAN + "\n* Ctrl-Cによる終了コマンドが入力されました。")
    print(Fore.CYAN + "* 現時点で学習済みモデルを保存しますか？")
    print(Fore.CYAN + "* [ Yes: (Y/y), No: (N/n) ] >>> ", end="")
    result = input()

    if result in ("Y", "y"):
        print(Fore.CYAN + "* ファイル名を入力してください。[デフォルト: model.pth] >>> ", end="")
        filename = input()
        filename = "model.pth" if filename == "" else filename
        save_model(model, filename)

    print(Fore.CYAN + "* プログラムを終了します。")
    sys.exit()
