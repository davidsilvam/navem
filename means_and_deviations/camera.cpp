#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <wiringPiI2C.h>
#include <stdlib.h>
#include <stdio.h>
#include <wiringPi.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h> 
#include <chrono>
#include <vector>
#include <sys/stat.h>

using namespace std;
using namespace cv;
using namespace chrono;

//DEFINIÇÕES MPU
#define Device_Address 0x68	/*Device Address/Identifier for MPU6050*/

#define PWR_MGMT_1   0x6B
#define SMPLRT_DIV   0x19
#define CONFIG       0x1A
#define GYRO_CONFIG  0x1B
#define INT_ENABLE   0x38
#define ACCEL_XOUT_H 0x3B
#define ACCEL_YOUT_H 0x3D
#define ACCEL_ZOUT_H 0x3F
#define GYRO_XOUT_H  0x43
#define GYRO_YOUT_H  0x45
#define GYRO_ZOUT_H  0x47

#define PI 3.14159265

/*
 * Classe MPU possuem todos os métodos para utilizar o sensor MPU na Rasp
 */
class MPU{
	public:
		float* Leituras();
		void MPU6050_Init();
		short Read_raw_data(int);
		MPU();
	private:
		int fd;
		//long long int inicio;
};

float* MPU::Leituras(){
	float Acc_x,Acc_y,Acc_z;
	float Gyro_x,Gyro_y,Gyro_z;
	//float Ax=0, Ay=0, Az=0;
	//float Gx=0, Gy=0, Gz=0;
	
	float* array = new float[7];
	
	Acc_x = Read_raw_data(ACCEL_XOUT_H);
	Acc_y = Read_raw_data(ACCEL_YOUT_H);
	Acc_z = Read_raw_data(ACCEL_ZOUT_H);
	
	Gyro_x = Read_raw_data(GYRO_XOUT_H);
	Gyro_y = Read_raw_data(GYRO_YOUT_H);
	Gyro_z = Read_raw_data(GYRO_ZOUT_H);
	
	//time_point<system_clock> now = system_clock::now();
    //array[7] = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
    
	
	/* Divide raw value by sensitivity scale factor */
	/*Ax = Acc_x/16384.0;
	Ay = Acc_y/16384.0;
	Az = Acc_z/16384.0;
	
	Gx = Gyro_x/131;
	Gy = Gyro_y/131;
	Gz = Gyro_z/131;*/

	array[0] = Acc_x/16384.0;
	array[1] = Acc_y/16384.0;
	array[2] = Acc_z/16384.0;
	
	array[3] = Gyro_x/131;
	array[4] = Gyro_y/131;
	array[5] = Gyro_z/131;
	
	//long long int ini = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();		
	//array[6] =  ini;
	
	//printf("\n Gx=%.3f °/s\tGy=%.3f °/s\tGz=%.3f °/s \n",array[3],array[4],array[5]);
	return array;
}

void MPU::MPU6050_Init(){	
	wiringPiI2CWriteReg8 (fd, SMPLRT_DIV, 0x07);	/* Write to sample rate register */
	wiringPiI2CWriteReg8 (fd, PWR_MGMT_1, 0x01);	/* Write to power management register */
	wiringPiI2CWriteReg8 (fd, CONFIG, 0);		/* Write to Configuration register */
	wiringPiI2CWriteReg8 (fd, GYRO_CONFIG, 24);	/* Write to Gyro Configuration register */
	wiringPiI2CWriteReg8 (fd, INT_ENABLE, 0x01);	/*Write to interrupt enable register */

} 

short MPU::Read_raw_data(int addr){
	short high_byte,low_byte,value;
	high_byte = wiringPiI2CReadReg8(fd, addr);
	low_byte = wiringPiI2CReadReg8(fd, addr+1);
	value = (high_byte << 8) | low_byte;
	return value;
}
	
MPU::MPU(void){
	fd = wiringPiI2CSetup(Device_Address);   /*Initializes I2C with device Address*/
	MPU6050_Init();/* Initializes MPU6050 */
	//time_point<system_clock> now = system_clock::now();
	//auto duration = now.time_since_epoch();
	//auto nanoseconds = duration_cast<microseconds>(duration);
	//inicio = inicioComum;
    //inicio = duration_cast<microseconds>(duration);
}

/*
 * Classe Camera possui todos os métodos para utilizar a câmera via USB na Rasp
 */
class Camera{
	public:
		Camera();
		void SalvarFrame(string, string);
	private:
		Mat frame;
		VideoCapture cap;	
};

/*
 * string path -> Caminho onde será salvo a imagem Ex: /home/img/
 * string nomeImg -> Nome de cada frame da imagem
 */
void Camera::SalvarFrame(string path, string nomeImg){
	if(!cap.open(0))
        return;     
	cap >> frame;	
	//imwrite(path + nomeImg + ".jpg" ,frame);
}

Camera::Camera(){
	cap.set(cv::CAP_PROP_FRAME_WIDTH,1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,720);
    cap.set(cv::CAP_PROP_FPS,30);
    cap.set(cv::CAP_PROP_EXPOSURE, -7);
}

/*
 * Classe Arquivo possui todos os métodos para salvar os dados dos sensores em arquivo .json
 */
class Arquivo{
	public:		
		void InicioJson(string);
		void FimJson();
		void ObjJson(double, double, double, long long int, bool);
		void ObjCameraJson(long long int,long long int, bool);
		void ObjTemposJson(long long int,bool);
		Arquivo(string, string);
	private:
		ofstream myfile;
};

/*
 * double v_x -> Valor do eixo x
 * double v_y -> Valor do eixo y
 * double v_z -> Valor do eixo z
 * int tempo_us -> Tempo em microsegundos
 * bool ultimo -> true, se for o último objeto a ser escrito.
 */
void Arquivo::ObjJson(double v_x, double v_y, double v_z, long long int tempo_us, bool ultimo = false){
	myfile << "{";
	myfile << "\"x\":" + to_string(v_x) + ",\n";
	myfile << "\"y\":" + to_string(v_y) + ",\n";
	myfile << "\"z\":" + to_string(v_z) + ",\n";
	myfile << "\"time_usec\":" + to_string(tempo_us) + "\n";
	if(ultimo){
		myfile << "}\n";
	}else{
		myfile << "},\n";
	}	
}

void Arquivo::ObjCameraJson(long long int tempoIni, long long int tempoFim, bool ultimo = false){
	myfile << "{\n";
	myfile << "\"time_ini_usec\":" + to_string(tempoIni) + ",\n";
	myfile << "\"time_fim_usec\":" + to_string(tempoFim) + "\n";
	myfile << "\"time_fim-ini_usec\":" + to_string(tempoFim - tempoIni) + "\n";
	if(ultimo){
		myfile << "}\n";
	}else{
		myfile << "},\n";
	}	
}

void Arquivo::ObjTemposJson(long long int TempoFrame, bool ultimo = false){
	myfile << "{";
	myfile << "\"Tempo de captura\":" + string("\"") + to_string(TempoFrame) + string("\"") + "\n";
	if(ultimo){
		myfile << "}\n";
	}else{
		myfile << "},\n";
	}	
}

/*
 * string sensor -> Nome do valor do objeto array no arquivo .json
 */
void Arquivo::InicioJson(string sensor){
	myfile << "{\n\"" + sensor + "\":[\n";
}

void Arquivo::FimJson(){
	myfile << "]\n}";
	myfile.close();
}

/*
 * string sensor -> Nome do valor do objeto array no arquivo .json
 */
Arquivo::Arquivo(string path, string nomeArq){
	myfile.open(path + nomeArq + ".json");
	InicioJson("dados");
}

double TemposGravacao(auto fim, auto inicio){
	double tempoDecorrido = std::chrono::duration_cast<std::chrono::microseconds>(fim - inicio).count();
	
	return tempoDecorrido;
	}

sem_t sen, cam;
pthread_mutex_t mutex1;
auto start = high_resolution_clock::now();

string data = __DATE__;
string hora = __TIME__;

//pasta = mkdir("/home/pi/Desktop/camera/" + data + ":" + hora,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
string dir = "/home/pi/Desktop/camera/" + data + ":" + hora;
int pasta = mkdir(dir.c_str() ,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
string path = dir + "/";
	
char flagSen = 0, flagCam = 0;
int tempoMilli = 5000;

long long int inicioComum = 0;
double tempos[100];
int quantidadeFrames = 10;

bool fimCamera = 0;

void* chamadaCamera(void* arg){
	 /*auto start = std::chrono::high_resolution_clock::now();
     imwrite("/home/pi/Pictures/imagem" + to_string(frameCount) + ".jpg" ,frame);
     auto end = std::chrono::high_resolution_clock::now();
     elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
     tempos[frameCount] = elapsedTime;*/
	//Camera camera;
	/*
	string data = __DATE__;
    string hora = __TIME__;
	string path = "/home/pi/Desktop/camera/Imgs/" + data + ":" + hora;
	*/
	Arquivo arqCamera(path, "camera");
	
	
	vector<Mat>frames;
	frames.reserve(100);
	
	VideoCapture cap(0);
	
	cap.set(cv::CAP_PROP_FRAME_WIDTH,1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT,720);
    //cap.set(cv::CAP_PROP_FPS,30);
	
	//VideoWriter video("outteste.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 10, Size(1280,720));
	
	int quantidadeFrames = 100;
	//while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() < tempoMilli)
	//printf("AGORAAAAAAAAAAAA !!!!!!!!!!!!!!!\n\n");
	for(int j = 0; j < quantidadeFrames ; j++)
	{
		Mat frame;
		/*if(!flagSen){
			sem_wait(&cam);
		}		
		pthread_mutex_lock(&mutex1);*/
		//camera.SalvarFrame(path, "img_" + to_string(i));
		//cout << ">>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
		
		long long int ini = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();		
		//
		auto start = std::chrono::high_resolution_clock::now();
		int aux = cap.read(frame);
		auto stop = std::chrono::high_resolution_clock::now();
		//
		long long int fim = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
		arqCamera.ObjCameraJson(ini, fim);
		//
		double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
		tempos[j] = elapsedTime;
		
		if (!aux){ 	
           cout << "deu ruim!";
           break;
		}
		
		frames.push_back(frame);
		//video.write(frame);
		//pthread_mutex_unlock(&mutex1);
		//if(!flagSen){
			//sem_post(&sen);
		//}	
		if(j == 99){
			fimCamera = 1;
		}
		
	}
	Arquivo arqTempo(path, "Tempos frame a frame");
	/*
	double soma = 0;
	double dp = 0;
    for(int i = 0;i<100;i++){
		arqTempo.ObjTemposJson(tempos[i]);
		cout << tempos[i] << endl;
        soma = soma + tempos[i];
        }
    double media = (soma/100);
    for(int j = 0; j < 100; j++){
        dp += pow((tempos[j] - media),2);
        }
    dp = dp/100;
    dp = sqrt(dp);
    cout << "media: " << media << endl;
    cout << "Desvio P.: " << dp << endl;
    
    arqTempo.FimJson();
	arqCamera.FimJson();
	* */
	printf("Gravando...");
	double tempoGravacao[quantidadeFrames];
	for(int i = 0; i < frames.size(); i++){
		
		auto inicio = std::chrono::high_resolution_clock::now();
		imwrite(path + to_string(i) + ".jpg" ,frames[i]); 
		auto fim = std::chrono::high_resolution_clock::now();
		//cout << TemposGravacao(fim,inicio) << endl;
		tempoGravacao[i] = TemposGravacao(fim,inicio);
		
	}
	
	double soma = 0;
	double dp = 0;
    for(int i = 0;i<100;i++){

		cout << tempoGravacao[i] << endl;
        soma = soma + tempoGravacao[i];
        
    }
        
    double media = (soma/100);
    for(int j = 0; j < 100; j++){
		
        dp += pow((tempoGravacao[j] - media),2);
        
    }
        
    dp = dp/100;
    dp = sqrt(dp);
    cout << "media: " << media << endl;
    cout << "Desvio P.: " << dp << endl;
    
	pthread_mutex_unlock(&mutex1);
	sem_post(&sen);
	flagCam = 1;
	printf("Finalizou thread camera\n");
	return NULL;
}

void* chamadaSensores(void* arg){
	MPU mpu;
	Arquivo arqGiro(path, "giroscopio");
	Arquivo arqAcel(path, "acelerometro");
	float* leituraSensores;
	//while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() < tempoMilli){
	while(fimCamera == 0){
		//if(!flagCam){
			//sem_wait(&sen);
		//}				
		//pthread_mutex_lock(&mutex1);
		//cout << "=================" << endl;
		leituraSensores = mpu.Leituras();
		long long int ini = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();		
		arqAcel.ObjJson(leituraSensores[0], leituraSensores[1], leituraSensores[2], ini);
		arqGiro.ObjJson(leituraSensores[3], leituraSensores[4], leituraSensores[5], ini);
		//pthread_mutex_unlock(&mutex1);
		//if(!flagCam){
			//sem_post(&cam);		
	}
	arqGiro.FimJson();
	arqAcel.FimJson();
	pthread_mutex_unlock(&mutex1);
	sem_post(&cam);
	flagSen = 1;
	printf("Finalizou thread sensores\n");
	return NULL;
}

int main(int argc, char *argv[]){
	//MPU mpu;
	//Camera camera;
	//string path = "/home/pi/Desktop/david/";
	/*while(1){
		mpu.Leituras();
	}*/
	/*for(int i = 0; i < 3; i++){
		camera.SalvarFrame(path, "img_" + to_string(i));
	}*/
	pthread_t chamadaCam, chamadaSen;
	sem_init(&cam, 0, 1);
	sem_init(&sen, 0, 0);
	pthread_mutex_init(&mutex1, NULL);
	//time_point<system_clock> now = system_clock::now();
	//printf("prepara\n\n");
	//usleep(5000000);
	//printf("start\n\n");
	inicioComum = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
	//printf("%ld\n\n", inicioComum);
	pthread_create(&chamadaCam, NULL, chamadaCamera, NULL);
	pthread_create(&chamadaSen, NULL, chamadaSensores, NULL);
	pthread_exit(NULL);
	return 0;
}
