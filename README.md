[[Paper](  )]
[[Paper (Long version)](  )]
[[Project Page]( )]
[[Demo](https://github.com/LongOPN/LongOPN/blob/main/AnonyModel.m4v)]
[[Colab](https://colab.research.google.com/drive/1HHDD_xp1NpododLkbIfXxWsT3AOYgb4n?usp=sharing)]
You can run the code and test any frames sequnces using provided colab demo

# ArrowGroup: Arrow Of Time Grouping 
![image](https://github.com/LongOPN/LongOPN/blob/main/LOPN2.jpg)
![image](https://github.com/LongOPN/LongOPN/blob/main/net89.jpg)

 
 


# UCF Data preparation:
Link to Google Drive to mount on colab:
 [[Link to UCF101 to be mounted on colab](https://drive.google.com/drive/folders/13zR2YhxZMGGTA_3kq9k3cEDJGf2xoCMP?usp=sharing)]


Note: to train on tuple longer than 4 frame you need to remove short samples.  You can apply provided related .txt in data folder

	├── UCF101/HMDB51
	│   ├── split
	│   │   ├── classInd.txt
	│   │   ├── testlist01.txt
	│   │   ├── trainlist01.txt
	│   │   └── ...
	│   └── video
	│       ├── ApplyEyeMakeup
	│       │   └── *.avi
	│       └── ...
	└── ...
