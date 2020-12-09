import glob
from PIL import Image
 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
 
if __name__ == "__main__":

    #gif画像で出力する
    #参考：https://own-search-and-study.xyz/2017/05/18/python%E3%81%AEmatplotlib%E3%81%A7gif%E3%82%A2%E3%83%8B%E3%83%A1%E3%82%92%E4%BD%9C%E6%88%90%E3%81%99%E3%82%8B/
        
    #画像ファイルの一覧を取得
    picList = glob.glob("result2/*.png")
        
    #figオブジェクトを作る
    fig = plt.figure()

    #軸を消す
    ax = plt.subplot(1, 1, 1)
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['left'].set_color('None')
    ax.spines['bottom'].set_color('None')
    ax.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
    ax.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')    

    #空のリストを作る
    ims = []
        
    #画像ファイルを順々に読み込んでいく
    for i in range(len(picList)):
            
        #1枚1枚のグラフを描き、appendしていく
        tmp = Image.open(picList[i])
        ims.append([plt.imshow(tmp)]) 
        
    #アニメーション作成    
    ani = animation.ArtistAnimation(fig, ims, interval=400, repeat_delay=10)
    ani.save("test.gif")