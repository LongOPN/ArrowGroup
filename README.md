  # ArrowGroup: Arrow Of Time Grouping 
![image](https://github.com/LongOPN/LongOPN/blob/main/LOPN2.jpg)
![image](https://github.com/LongOPN/LongOPN/blob/main/net89.jpg)

 
[[Paper](  )]
[[Paper (Long version)](  )]
[[Project Page]( )]
[[Demo](https://github.com/LongOPN/LongOPN/blob/main/AnonyModel.m4v)]
[[Colab](https://colab.research.google.com/drive/1HHDD_xp1NpododLkbIfXxWsT3AOYgb4n?usp=sharing)]



# UCF Data preparation:
Link to Google Drive to mount on colab:
 [[Link to UCF101 to be mount on colab](https://drive.google.com/drive/folders/13zR2YhxZMGGTA_3kq9k3cEDJGf2xoCMP?usp=sharing)]


Note: to train on tuple longer than 4 frame you need to remove apply provided .txt in data folder

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
