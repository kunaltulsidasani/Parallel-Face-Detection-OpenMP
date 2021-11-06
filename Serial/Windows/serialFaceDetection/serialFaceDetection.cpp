/* Face Detection on Photos - Parallel Computing Project - Serial Code
Team - 
Kunal 2019085
Chirag Agrawal 2019233
Bhomik Sharma 2019226
Vasukumar Kotadiya 2019171
 */

//Include libraries needed
#include<iostream>
#include<omp.h>
#include<stdlib.h>
#include<opencv2/objdetect.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

#ifdef _WIN32
#include"dirent.h"
#endif

#ifdef linux
#include<dirent.h>
#endif

#include<sys/types.h>

using namespace std;
using namespace cv;


int main(){
    //Declare a object of CascadeClassifier class which will load the data for face detection, which is already present in openCV  
    CascadeClassifier faceDetection;

    //Load the data file
    //if not loaded the function will exit
    if(!faceDetection.load("haarcascade_frontalface_default.xml")){
        cout<<"Data file not loaded \n";
        exit(0);
    }

    double run_time = 0;

    const char path[] = "Input";
    struct dirent *entry;
    
    //Open Input dir
    DIR *dir = opendir(path);
   
    //if no such dir exists exit code
    if (dir == NULL){
        cout <<"input directory missing\n";
        return 0;
    }

    //Begin and end time
    double t1, t2;

    //image names in input folder to detect faces
    vector<string> img_names;
    while ((entry = readdir(dir)) != NULL) {
        img_names.push_back(entry->d_name);
    }
    
    //Detect faces in all the given files in Input directory
    int N = img_names.size();
    
    //Begin time
    t1 = omp_get_wtime(); 

    for(int i=0;i<N;i++)
    {
        //image path
        string img_path = path;
        img_path += "/";
        img_path += img_names[i];

        //take the input image through imread function, without any changes to image
        Mat img = imread(img_path, IMREAD_UNCHANGED);

        //if the image is not loaded the program will exit
        if(img.empty()){
            continue;
        }
        else{
            // cout<<"image loaded \t"<<img_path<<"\n";

            //Vector object where points of square around detected faces will be stored.
            vector<Rect> faces;

            //function for face detection and all the face detected will be stored in vector faces
            faceDetection.detectMultiScale(img, faces);

            //Loop to draw rectangles on all detected faces
            for(int i=0; i<faces.size(); i++){
                Point pt1(faces[i].x, faces[i].y);
                Point pt2((faces[i].x + faces[i].height), (faces[i].y + faces[i].width));
                rectangle(img,pt1,pt2,Scalar(0, 0, 255), 2, 8, 0);
            }
            
            //name of output image
            string out = "Output/out_"; 
            out += img_names[i];
            
            //Save image
            imwrite(out, img);
            cout<<out+"\n";
        }   
    }
    //end time
    t2 = omp_get_wtime();

    //Total run time
    run_time = (t2 - t1);
    
    //Close Open Directory
    closedir(dir);
    cout<<"serial face detection time = "<<run_time<<"\n";
    
    return 0;
}