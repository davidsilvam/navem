// spi.c
//
// Example program for bcm2835 library
// Shows how to interface with SPI to transfer a byte to and from an SPI device
//
// After installing bcm2835, you can build this 
// with something like:
// gcc -o spi spi.c -l bcm2835
// sudo ./spi
//
// Or you can test it before installing with:
// gcc -o spi -I ../../src ../../src/bcm2835.c spi.c
// sudo ./spi
//
// Author: Mike McCauley
// Copyright (C) 2012 Mike McCauley
// $Id: RF22.h,v 1.21 2012/05/30 01:51:25 mikem Exp $
 
#include <bcm2835.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <string>

using namespace std;
 
int main(int argc, char **argv)
{
    // If you call this, it will not actually access the GPIO
// Use for testing
//        bcm2835_set_debug(1);
 
    if (!bcm2835_init())
    {
      printf("bcm2835_init failed. Are you running as root??\n");
      return 1;
    }
 
    if (!bcm2835_spi_begin())
    {
      printf("bcm2835_spi_begin failed. Are you running as root??\n");
      return 1;
    }
    bcm2835_spi_setBitOrder(BCM2835_SPI_BIT_ORDER_MSBFIRST);      // The default
    bcm2835_spi_setDataMode(BCM2835_SPI_MODE0);                   // The default
    bcm2835_spi_setClockDivider(BCM2835_SPI_CLOCK_DIVIDER_1024); // The default
    bcm2835_spi_chipSelect(BCM2835_SPI_CS0);                      // The default
    bcm2835_spi_setChipSelectPolarity(BCM2835_SPI_CS0, LOW);      // the default
    
    // Send a byte to the slave and simultaneously read a byte back from the slave
    // If you tie MISO to MOSI, you should read back what was sent
    char buf [40];
    
    ofstream myfile;
    myfile.open("./leituras_arduino/parado/" + string(argv[1]) + ".txt");

    
    vector<string> leituras;
    leituras.reserve(20000);
    unsigned long int cont = 0;            
    uint8_t read_data;
    
    /*cout << "Iniciou primeiro" << endl;
    
    while(1){
        read_data = bcm2835_spi_transfer(1);
        for (int pos = 0; pos < sizeof (buf) - 1; pos++){
            delayMicroseconds (15);
            buf [pos] = bcm2835_spi_transfer (0);
            if (buf [pos] == 0){
                break;
            }
        } 
        usleep(100000);
        cout << buf << endl;
    }
    
    cout << buf << endl;
    cout << "Terminou primeiro" << endl;*/
    
      //Preparacao
    cout << "Inicia em 5 segundos..." << endl;
    usleep(1000000);
    cout << "4 segundos" << endl;
    usleep(1000000);
    cout << "3 segundos" << endl;
    usleep(1000000);
    cout << "2 segundos" << endl;
    usleep(1000000);
    cout << "1 segundos" << endl;
    usleep(1000000);
    cout << "Iniciou!" << endl;
    
    auto time_ini = std::chrono::high_resolution_clock::now();
    while(1){     
        read_data = bcm2835_spi_transfer(2); 
        //cout << read_data << endl;   
        //printf("%x\n", read_data) ;
        if( read_data == 1){        
            auto start = std::chrono::high_resolution_clock::now();
            read_data = bcm2835_spi_transfer(1);
            for (int pos = 0; pos < sizeof(buf) - 1; pos++){
                delayMicroseconds (100); //15
                buf [pos] = bcm2835_spi_transfer(0);
                if (buf [pos] == 0){
                    break;
                }
            }
            auto stop = std::chrono::high_resolution_clock::now();
            double elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(stop-start).count();
            double elapsedTimeWalking = std::chrono::duration_cast<std::chrono::microseconds>(stop-time_ini).count();
            leituras.push_back(buf);
            cout << chrono::duration_cast<std::chrono::microseconds>(start.time_since_epoch()).count() << " " << chrono::duration_cast<std::chrono::microseconds>(stop.time_since_epoch()).count() << " " << elapsedTime << " - " << buf << endl;
            //if(cont++ > 20) break;
            cont++;
            if(elapsedTimeWalking > 10000000) break;
        }
        //usleep(200000);
    }

    cout << "Gravando leituras..." << endl;
    for(int j = 0; j < cont;j++){
        //cout << leituras[j] << endl;
        myfile << leituras[j] << "\n";
    }
    myfile.close();
    
    cout << "Terminou!" << endl;

    bcm2835_spi_end();
    bcm2835_close();
    return 0;
}
