#ifndef __BPPERCEPTRON_H__
#define __BPPERCEPTRON_H__

#include<vector>
using namespace std;

class BPPerceptron{
private:
	vector<vector<double>> weightInput;//����㵽���������Ȩ
	vector<vector<double>> weightMid;//���㵽����������Ȩ
	vector<double> thresholdMid;//�������ֵ
	vector<double> thresholdOutput;//��������ֵ
	vector<vector<double>> sampleData;//������������
	double learningRate;//ѧϰ�ʼ�����
	double errorRate;//������
	int maxIterateTime;//�趨������������
	int iterateTime;//���յ�������

	double getRand();//��ȡ0-1�����������
	vector<vector<double>> getFuncRes(vector<double> sample);//��ȡ����ģ�ͺ����������(0)���������(1)
	
public:

	BPPerceptron();//�޲ι��캯��

	BPPerceptron(int inputNum, int midNum, int outputNum, double learnRate, int iterateTimes);//����BPģ����

	void randomParameter();//���������
	
	void loadFile(char *file, int totaline);//���������ļ�
	
	void trainModel();//BP�㷨ѵ��ģ��

	double calErroRate();//��������������
	
	void printSampleData();//��ӡ��������

	void printParameter();//��ӡ����

	int getTrainTimes();//�����������
	
};

#endif
