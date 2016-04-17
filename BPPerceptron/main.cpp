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
		cout << "   ����������ڵ���Ŀ��ѧϰ�ʣ� " << endl;
		cout << "   ";
		cin >> m >> n;
		BPPerceptron bp(2, m, 2, n, 5000);
		cout << endl << "****************************************" << endl;
		cout << "   ��ѡ������ļ� : 1.ls.csv  2.xor.csv 3.�˳�   : " ;
		
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
			cout << "   �������󣬳����Զ��˳���" << endl;
		}
		
		bp.trainModel();
		cout << endl << "****************************************" << endl;
		cout << "   ����ѵ���󣬴�����Ϊ�� ";
		cout << bp.calErroRate() << endl;
		cout << endl<< "   ѵ����������Ϊ�� " << bp.getTrainTimes() << endl;
		cout << endl << "****************************************" << endl;
		cout << "   �����������: " << endl;
		bp.printParameter();
		cout << endl << "****************************************" << endl;
	}
}