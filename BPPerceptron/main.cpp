#include<iostream>
#include"BPPerceptron.h"
using namespace std;



int main()
{
	
	char *ls = "ls.csv";
	char *xor = "xor.csv";
	while (1)
	{
		double m; double n;
		cout << endl << "****************************************" << endl;
		cout << "   请输入隐层节点数目与学习率： " << endl;
		cout << "   ";
		cin >> m >> n;
		BPPerceptron bp(2, m, 2, n, 5000);
		cout << endl << "****************************************" << endl;
		cout << "   请选择操作文件 : 1.ls.csv  2.xor.csv 3.退出   : " ;
		
		int index = 0; 
		cin >> index;
		cout << endl << "****************************************" << endl;
		
		
		if (index == 1)
		    bp.loadFile(ls, 600);
		else if (index == 2)
			bp.loadFile(xor, 800);
		else if (index == 3)
			return 0;
		else{
			cout << "   输入有误，程序自动退出！" << endl;
		}
		
		bp.trainModel();
		cout << endl << "****************************************" << endl;
		cout << "   经过训练后，错误率为： ";
		cout << bp.calErroRate() << endl;
		cout << endl<< "   训练迭代次数为： " << bp.getTrainTimes() << endl;
		cout << endl << "****************************************" << endl;
		cout << "   输出参数如下: " << endl;
		bp.printParameter();
		cout << endl << "****************************************" << endl;
	}
}