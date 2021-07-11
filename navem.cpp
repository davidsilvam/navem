      #include <ctime>
      #include <fstream>
      #include <iostream>
      #include <unistd.h>
      #include <raspicam/raspicam.h>
      #include "opencv2/opencv.hpp"
      
      #include <math.h>
      #include <pthread.h>
      #include <semaphore.h> 
      #include <chrono>
      #include <vector>
      #include <sys/stat.h>
      #include <string>
      #include <stdio.h>
      
      #include <bcm2835.h>
      
      using namespace std;
      using namespace cv;
      using namespace chrono;
      unsigned int calibration = 0;
      unsigned const int qtdFrames = 100; //MAX: 700
      
      void media_e_desvioPadrao(int* vector[], int size);
      
      /*
      * Classe Camera possui todos os métodos para utilizar a câmera via USB na Rasp
      */
      class Camera{
            public:
                  Camera();
                  void SalvarFrame(string, string);
                  void GetImage();
                  raspicam::RaspiCam CameraObj; //Cmaera object
                  unsigned char *data;
                  int w = 640;//1280
                  int h = 480;//960
            private:
                  Mat frame;                
            //VideoCapture cap;	
      };
      
      Camera::Camera(){
            cout << "VALORES PADROES CAMERA: " << endl;
            cout << "ISO: " << CameraObj.getISO() << endl;
            cout << "Shutter Speed" << CameraObj.getShutterSpeed() << endl;
            cout << "Exposure:" << CameraObj.getExposure() << endl;
            
            CameraObj.setCaptureSize(w, h);
            CameraObj.setFrameRate(6);
            CameraObj.setShutterSpeed(20000); //  1/125 // 8000 // 500
            //CameraObj.setExposureCompensation(0);
            //CameraObj.setBrightness(70);
            //CameraObj.setSharpness (0);
            //CameraObj.setContrast (15);
            //CameraObj.setSaturation (15);
            //CameraObj.setISO (400);  // 400/800 *
            //CameraObj.setVideoStabilization(true);
            //CameraObj.setExposureCompensation (25); // *
            CameraObj.setExposure (raspicam::RASPICAM_EXPOSURE_AUTO);
            //CameraObj.setAWB(raspicam::RASPICAM_AWB_AUTO);
            //CameraObj.setAWB_RB(1, 1);
            
            //Open camera 
            cout << "Opening Camera..." << endl;
            if ( !CameraObj.open()) {cerr << "Error opening camera" << endl;}
            //wait a while until camera stabilizes
            cout << "Sleeping for 3 secs" << endl;
            //cv::Mat image = cv::Mat::zeros(960,1280,CV_8UC3);
            //usleep(3);
      }
      
      /*
      * string path -> Caminho onde será salvo a imagem Ex: /home/img/
      * string nomeImg -> Nome de cada frame da imagem
      */
      void Camera::SalvarFrame(string path, string nomeImg){
            CameraObj.grab();
            data = new unsigned char[CameraObj.getImageTypeSize (raspicam::RASPICAM_FORMAT_RGB)];
            int width = 1280*3;
            int height = 720;
            
            cv::Mat image = cv::Mat(720,1280,CV_8UC3, data);
            cvtColor(image, image, 4);
            
            cv::imwrite("image.jpg",image );
      }
      
      void Camera::GetImage(){
            CameraObj.grab();
            data = new unsigned char[CameraObj.getImageTypeSize (raspicam::RASPICAM_FORMAT_RGB)];    
            cv::Mat image = cv::Mat(720,1280,CV_8UC3, data);
            CameraObj.retrieve ( data,raspicam::RASPICAM_FORMAT_RGB );//get camera image
            cvtColor(image, image, 4);
            
            cv::imwrite("image.jpg",image );
            
            //return cv::Mat(960,1280,CV_8UC3, data);
      }
      
      /*
      * Classe Arquivo possui todos os métodos para salvar os dados dos sensores em arquivo .json
      */
      class Arquivo{
            public:		
            void InicioJson(string);
            void FimJson();
            void ObjJson(string, string, string, string, int, bool);
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
      void Arquivo::ObjJson(string v_x, string v_y, string v_z, string tempo_us, int index, bool ultimo = false){
            myfile << "{";
            myfile << "\"index\":" + to_string(index) + ",\n";
            myfile << "\"AccX\":" + v_x + ",\n";
            myfile << "\"AccY\":" + v_y + ",\n";
            myfile << "\"AccZ\":" + v_z + ",\n";
            myfile << "\"time_sec\":" + tempo_us + "\n";
            if(ultimo){
            myfile << "}\n";
            }else{
            myfile << "},\n";
      }	
      }
      
      void Arquivo::ObjCameraJson(long long int tempoIni, long long int tempoFim, bool ultimo = false){
            myfile << "{\n";
            myfile << "\"time_ini_usec\":" + to_string(tempoIni) + ",\n";
            myfile << "\"time_fim_usec\":" + to_string(tempoFim) + ",\n";
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
      
      string dir = "/home/pi/Desktop/experimentos/Exps_10.07.2021/" + data + ":" + hora;
      int pasta = mkdir(dir.c_str() ,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      string path = dir + "/";
      
      char flagSen = 0, flagCam = 0;
      int tempoMilli = 5000;
      
      long long int inicioComum = 0;
      double tempos[qtdFrames];
      
      bool fimCamera = 0;
      
void* chamadaCamera(void* arg){
      Arquivo arqCamera(path, "frames");
      Camera camera;    
      vector<Mat> frames;
      frames.reserve(qtdFrames);
      printf("Iniciando câmera\n\n");
      while(calibration){}
      
      cout << "\n" << "INICIO" << endl;
      for(int j = 0; j < qtdFrames ; j++){
            cout << "j: " << j << endl;
            Mat frame;
            
            long long int ini = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();		
            //
            auto start = std::chrono::high_resolution_clock::now();
            camera.CameraObj.grab();
            auto stop = std::chrono::high_resolution_clock::now();
            //
            long long int fim = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            arqCamera.ObjCameraJson(ini, fim, j == qtdFrames - 1);
            //
            double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
            tempos[j] = elapsedTime;
            
            camera.data = new unsigned char[camera.CameraObj.getImageTypeSize (raspicam::RASPICAM_FORMAT_RGB)];       
            camera.CameraObj.retrieve (camera.data);
            
            frames.push_back(cv::Mat(camera.h,camera.w,CV_8UC3, camera.data));
            
      }
      fimCamera = 1;
      double soma = 0;
      double dp = 0;
      for(int i = 0;i<qtdFrames;i++){	
            soma = soma + tempos[i];        
      }
      
      double media = (soma/qtdFrames);
      for(int j = 0; j < qtdFrames; j++){		
            dp += pow((tempos[j] - media),2);        
      }
      
      dp = dp/qtdFrames;
      dp = sqrt(dp);
      cout << "media dos tempos entre frames em us: " << media << endl;
      cout << "Desvio P. dos tempos entre frames: " << dp << endl;
      
      Arquivo arqTempo(path, "Tempos frame a frame");
      
      printf("Gravando...\n");
      double tempoGravacao[qtdFrames];
      for(int i = 0; i < frames.size(); i++){
            cvtColor(frames[i], frames[i], 4);
            if(i < 10)
                  imwrite(path + "0" + to_string(i) + ".jpg" ,frames[i]); 
            else  
                  imwrite(path + to_string(i) + ".jpg" ,frames[i]); 
      
      }
      arqTempo.FimJson();
      arqCamera.FimJson();
      pthread_mutex_unlock(&mutex1);
      sem_post(&sen);
      flagCam = 1;
      printf("Finalizou thread camera\n");
      return NULL;
  

      }
      
void* chamadaSensores(void* arg){
      //MPU mpu;
      Arquivo arqGiro(path, "rotations");
      Arquivo arqAcel(path, "accelerations");
      
      if (!bcm2835_init()){
            printf("bcm2835_init failed. Are you running as root??\n");
            //return 1;
      }
      
      if (!bcm2835_spi_begin()){
            printf("bcm2835_spi_begin failed. Are you running as root??\n");
            //return 1;
      }
      
      bcm2835_spi_setBitOrder(BCM2835_SPI_BIT_ORDER_MSBFIRST);      // The default
      bcm2835_spi_setDataMode(BCM2835_SPI_MODE0);                   // The default
      bcm2835_spi_setClockDivider(BCM2835_SPI_CLOCK_DIVIDER_1024); // The default
      bcm2835_spi_chipSelect(BCM2835_SPI_CS0);                      // The default
      bcm2835_spi_setChipSelectPolarity(BCM2835_SPI_CS0, LOW);      // the default
      
      // Send a byte to the slave and simultaneously read a byte back from the slave
      // If you tie MISO to MOSI, you should read back what was sent
      char buf [49];  
      
      vector<string> leituras;
      leituras.reserve(20000);
      unsigned long int contador = 0;       
           
      uint8_t read_data; 
      bcm2835_spi_transfer(1); // INICIA CALIBRAÇÃO
      
      calibration = (int)bcm2835_spi_transfer(2);
      
      while(calibration){ //VERIFICA SE AINDA ESTÁ CALIBRANDO
            calibration = (int)bcm2835_spi_transfer(2);
            cout << "Calibração " << calibration << endl;
            usleep(100000);
            contador++;
      }
      
      contador = 0;
      
      while(fimCamera == 0){
            read_data = bcm2835_spi_transfer(3);  
            
            usleep(50000);
            
            for (int pos = 0; pos < sizeof(buf) - 1; pos++){
                    delayMicroseconds (30);
                    buf [pos] = bcm2835_spi_transfer(0);
                        
                    if (buf [pos] == 0){
                         break;
                    }
            }   
            
            
            cout << "i " << contador << ": ";
            cout << buf << endl;
            contador++;     

            leituras.push_back(buf);
            pthread_mutex_unlock(&mutex1);
            if(!flagCam)
            sem_post(&cam);	
            //}
      }
      
      string accx,accy,accz,row,pitch,yaw,tempos;
      unsigned int contadorDoisPontos = 0;     
      cout << endl;
      for(int i = 0; i < leituras.size(); i++){
            contadorDoisPontos = 0;
            accx = "",accy = "",accz = "",row = "",pitch = "",yaw = "",tempos = "";
            
            for(int j = 1; j < leituras[i].size(); j++){
                  
                    if(leituras[i][j] == ':'){
                          contadorDoisPontos++;
                          continue;
                    }
                    else if(leituras[i][j] == '\0'){
                        cout << "VEIO VAZIO" << endl;
                        break;
                    }
                    else{
                          if(contadorDoisPontos == 0) accx += leituras[i][j];
                          else if(contadorDoisPontos == 1) accy += leituras[i][j];
                          else if(contadorDoisPontos == 2) accz += leituras[i][j];
                          else if(contadorDoisPontos == 3) row += leituras[i][j];
                          else if(contadorDoisPontos == 4) pitch += leituras[i][j];
                          else if(contadorDoisPontos == 5) yaw += leituras[i][j];
                          else if(contadorDoisPontos == 6) tempos += leituras[i][j];
                    }
                    
            }
            /*
        cout << i << ": ";
	  cout << "AccX: " << accx << endl;
	  cout << "AccY: " << accy << endl;
	  cout << "AccZ: " << accz << endl;
	  cout << "Row: " << row << endl;
	  cout << "Pitch: " << pitch << endl;
	  cout << "Yaw: " << yaw << endl;
	  cout << "Tempo " << tempos << endl;
        
        cout << endl;*/
        arqAcel.ObjJson(accx,accy,accz,tempos, i, i == (leituras.size() - 1));
      }
      
      

      arqGiro.FimJson();
      arqAcel.FimJson();
      pthread_mutex_unlock(&mutex1);
      sem_post(&cam);
      flagSen = 1;
      printf("Finalizou thread sensores\n");
      return NULL;
      
      }
      
int main ( int argc,char **argv ) {
            pthread_t chamadaSen, chamadaCam;
           
            sem_init(&cam, 0, 1);
            sem_init(&sen, 0, 0);
            
            pthread_mutex_init(&mutex1, NULL);
            
            inicioComum = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
            
            pthread_create(&chamadaSen, NULL, chamadaSensores, NULL);
            pthread_create(&chamadaCam, NULL, chamadaCamera, NULL);
            
            pthread_exit(NULL);
            return 0;
      }
      
void media_e_desvioPadrao(int* vector[], int size){
             double soma = 0;
             double dp = 0;

             for(int i = 0;i< size;i++){			
                  soma = soma + *vector[i];        
             }
            
             double media = (soma/size);
             for(int j = 0; j < size; j++){		
                  dp += pow((*vector[j] - media),2);        
             } 
            
             dp = dp/size;
             dp = sqrt(dp);
            
             cout << "Média: " << media << endl;
             cout << "Desvio padrão: " << dp << endl;
      }
      
      
