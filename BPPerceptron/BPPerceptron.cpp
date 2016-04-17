#include"BPPerceptron.h"
#include<fstream>
#include<iostream>
#include <cstdlib>
#include <ctime>
#include<math.h>
double BPPerceptron::getRand()
{
		//srand(unsigned int(time(NULL)));
		double s = (rand() + 500) % (1000) / (double)(1000);
		return s;
};

vector<vector<double>> BPPerceptron::getFuncRes(vector<double> sample)
{
		vector<double> midOutput;
		for (unsigned int i = 0; i < thresholdMid.size(); i++)
		{
			double input = 0;
			for (unsigned int j = 0; j < weightInput.size(); j++)
			{
				input += sample[j] * weightInput[j][i];
			}
			input -= thresholdMid[i];
			double output = 0;
			output = double(1) / double(1 + exp(-input));
			midOutput.push_back(output);
		}


		vector<double> Output;
		for (unsigned int i = 0; i < thresholdOutput.size(); i++)
		{
			double input = 0;
			for (unsigned int j = 0; j < weightMid.size(); j++)
				input += midOutput[j] * weightMid[j][i];
			input -= thresholdOutput[i];
			double output = 0;
			output = double(1) / double(1 + exp(-input));
			Output.push_back(output);
		}

		vector<vector<double>> finalOut;
		finalOut.push_back(Output);
		finalOut.push_back(midOutput);
		return finalOut;
};


BPPerceptron::BPPerceptron()
{
		learningRate = 0.1;
		maxIterateTime = 5000;
};

BPPerceptron::BPPerceptron(int inputNum, int midNum, int outputNum, double learnRate, int iterateTimes)
{
		for (int i = 0; i < inputNum; i++)
		{
			vector<double> weightIn;
			for (int j = 0; j < midNum; j++)
				weightIn.push_back(getRand());
			weightInput.push_back(weightIn);
		}

		for (int i = 0; i < midNum; i++)
		{
			vector<double> weightMedium;
			for (int j = 0; j < outputNum; j++)
				weightMedium.push_back(getRand());
			weightMid.push_back(weightMedium);
		}

		for (int i = 0; i < midNum; i++)
			thresholdMid.push_back(getRand());

		for (int i = 0; i < outputNum; i++)
			thresholdOutput.push_back(getRand());

		learningRate = learnRate;
		iterateTime = iterateTimes; 
		maxIterateTime = 5000;

};

void BPPerceptron::randomParameter()
{
		for (unsigned int i = 0; i < weightInput.size(); i++)
		{

			for (unsigned int j = 0; j < weightInput[i].size(); j++)
				weightInput[i][j] = getRand();

		}

		for (unsigned int i = 0; i <weightMid.size(); i++)
		{

			for (unsigned int j = 0; j < weightMid[i].size(); j++)
				weightMid[i][j] = (getRand());

		}

		for (unsigned int i = 0; i < thresholdMid.size(); i++)
			thresholdMid[i] = (getRand());

		for (unsigned int i = 0; i < thresholdOutput.size(); i++)
			thresholdOutput[i] = (getRand());
};

void BPPerceptron::loadFile(char *file, int totaline)
{
		ifstream input(file);
		int nowline = 0;
		if (!input)
		{
			cout << "输入数据文件不存在" << endl;
			exit(-1);
		}
		if (weightInput.size() < 1)
		{
			cout << "类未被正确构造，无法载入数据" << endl;
			return;
		}
		while (!input.eof())
		{
			vector<double> line;
			if (nowline >= totaline)
			{
				input.close();
				return;
			}
			nowline++;
			int attriNum = weightInput.size();

			while (attriNum > 0)
			{
				double value = 0;
				input >> value;
				char spilt;
				input >> spilt;

				line.push_back(value);
				attriNum--;

			}
			int catagory = 0;
			input >> catagory;

			if (catagory == 1)
				line.push_back(0);
			else
				line.push_back(1);
			sampleData.push_back(line);
		};
		cout << sampleData.size() << endl;
		input.close();
};

void BPPerceptron::trainModel()
{
		randomParameter();
		int nowtime = 0;
		while (nowtime<maxIterateTime){
			nowtime++;
			for (unsigned int m = 0; m < sampleData.size(); m++){


				double catagory = sampleData[m][sampleData[m].size() - 1];
				vector<double> knownOut;
				knownOut.push_back(catagory);
				knownOut.push_back(1 - catagory);
				vector<double> modelOut = getFuncRes(sampleData[m])[0];
				vector<double> b = getFuncRes(sampleData[m])[1];
				vector<double> g;

				for (unsigned int i = 0; i < thresholdOutput.size(); i++)
				{
					double gj = modelOut[i] * (1 - modelOut[i])*(knownOut[i] - modelOut[i]);
					g.push_back(gj);
				}

				vector<double> e;
				for (unsigned int h = 0; h < thresholdMid.size(); h++)
				{
					double temp = 0;
					for (unsigned int j = 0; j < thresholdOutput.size(); j++)
					{
						temp += weightMid[h][j] * g[j];
						weightMid[h][j] += learningRate*g[j] * b[h];
					}
					double eh = b[h] * (1 - b[h])*temp;
					e.push_back(eh);
				}

				for (unsigned int i = 0; i < weightInput.size(); i++)
					for (unsigned int h = 0; h < thresholdMid.size(); h++)
					{
						weightInput[i][h] += learningRate * e[h] * sampleData[m][i];
					}

				for (unsigned int i = 0; i < thresholdOutput.size(); i++)
					thresholdOutput[i] += -1 * learningRate*g[i];

				for (unsigned int h = 0; h < thresholdMid.size(); h++)
					thresholdMid[h] += -1 * learningRate*e[h];



			}

			double errRate = calErroRate();
			cout << "   第 " << nowtime << " 次迭代，错误率为 : " << errRate << endl;
			if (abs(errRate) <= 0.00001)
			{
				iterateTime = nowtime;
				return;
			}
			
		}

};

double BPPerceptron::calErroRate()
{
		int errorNum = 0;
		for (unsigned int m = 0; m < sampleData.size(); m++){
			double catagory = sampleData[m][sampleData[m].size() - 1];
			vector<double> knownOut;
			knownOut.push_back(catagory);
			knownOut.push_back(1 - catagory);
			vector<double> modelOut = getFuncRes(sampleData[m])[0];

			for (unsigned int i = 0; i < thresholdOutput.size(); i++)
			{
				
				if (abs(modelOut[i] - knownOut[i]) >= 0.5)
				{
					errorNum++;
					break;
				}

			}
		}
		errorRate = (double)errorNum / double(sampleData.size());
		return errorRate;

};

void BPPerceptron::printSampleData()
{
		for (unsigned int i = 0; i < sampleData.size(); i++)
		{
			for (unsigned int j = 0; j < sampleData[i].size(); j++)
				cout << sampleData[i][j] << " ";
			cout << endl;
		}
};

void BPPerceptron::printParameter()
{
		cout << endl << "----------------------------------------" << endl;
		cout << "  InputNum : " << weightInput.size() << "   MidNum : " << thresholdMid.size() << "     OutputNum : " << thresholdOutput.size() << endl;
		cout << "  LearningRate : " << learningRate << endl;

		cout << endl << "----------------------------------------" << endl;
		cout << "  Weight Input To Mid :" << endl << endl;
		for (unsigned int i = 0; i < weightInput.size(); i++)
		{
			for (unsigned int j = 0; j < weightInput[i].size(); j++)
				cout << "    " << weightInput[i][j];
			cout << endl;
		}

		cout << endl << "----------------------------------------" << endl;
		cout << "  Threshold For Mid : " << endl << endl;
		for (unsigned int i = 0; i < thresholdMid.size(); i++)
			cout << "   " << thresholdMid[i];
		cout << endl;

		cout << endl << "----------------------------------------" << endl;
		cout << "  Weight Mid To Output :" << endl << endl;
		for (unsigned int i = 0; i < thresholdMid.size(); i++)
		{
			for (unsigned int j = 0; j < weightMid[i].size(); j++)
				cout << "  " << weightMid[i][j];
			cout << endl;
		}

		cout << endl << "----------------------------------------" << endl;
		cout << "  Threshold For Output : " << endl << endl;
		for (unsigned int i = 0; i < thresholdOutput.size(); i++)
			cout << "   " << thresholdOutput[i];
		cout << endl;
		cout << endl << "----------------------------------------" << endl;
};

int BPPerceptron::getTrainTimes(){ return iterateTime; };