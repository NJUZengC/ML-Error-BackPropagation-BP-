#ifndef __BPPERCEPTRON_H__
#define __BPPERCEPTRON_H__

#include<vector>
using namespace std;

class BPPerceptron{
private:
	vector<vector<double>> weightInput;//输入层到隐层的连接权
	vector<vector<double>> weightMid;//隐层到输出层的连接权
	vector<double> thresholdMid;//隐层的阈值
	vector<double> thresholdOutput;//输出层的阈值
	vector<vector<double>> sampleData;//输入样本数据
	double learningRate;//学习率即步长
	double errorRate;//错误率
	int maxIterateTime;//设定的最大迭代次数
	int iterateTime;//最终迭代次数

	double getRand();//获取0-1的随机浮点数
	vector<vector<double>> getFuncRes(vector<double> sample);//获取套入模型后的输出层输出(0)和隐层输出(1)
	
public:

	BPPerceptron();//无参构造函数

	BPPerceptron(int inputNum, int midNum, int outputNum, double learnRate, int iterateTimes);//建立BP模型类

	void randomParameter();//随机化参数
	
	void loadFile(char *file, int totaline);//加载数据文件
	
	void trainModel();//BP算法训练模型

	double calErroRate();//计算和输出错误率
	
	void printSampleData();//打印测试数据

	void printParameter();//打印参数

	int getTrainTimes();//输出迭代次数
	
};

#endif
