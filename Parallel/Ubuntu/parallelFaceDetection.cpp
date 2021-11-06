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


    //Set number of threads as 8
    omp_set_num_threads(8);

    //Detect faces in all the given files in Input directory
    int N = img_names.size();

    //Thread ID and Number of threads
    int tid, numt, i;

    //Begin time
    t1 = omp_get_wtime();
    #pragma omp parallel default(shared) private(tid, numt, i)
    {
        //Declare a object of CascadeClassifier class which will load the data for face detection, which is already present in openCV
        CascadeClassifier FD;

        //Load the data file
        //if not loaded the function will exit
        if(!FD.load("haarcascade_frontalface_default.xml")){
            cout<<"No load file\n";
            exit(0);
        }

        int from, to;

        tid = omp_get_thread_num();
        numt = omp_get_num_threads();

        from = (N/numt)*tid;
        to = (N/numt)*(tid+1) - 1;

        if(tid == numt -1){
            to = N-1;
        }

        for(i=from;i<=to;i++){
            
            //imgae path
            string img_path = path;
            img_path += "/";
            img_path += img_names.at(i);

            //take the input image through imread function, without any changes to image
            Mat img = imread(img_path, IMREAD_UNCHANGED);

            //if the image is not loaded the program will exit
            if(img.empty()){
                continue;
            }
            else{
                //Vector object where points of square around detected faces will be stored.
                vector<Rect> faces;

                //function for face detection and all the face detected will be stored in vector faces
                FD.detectMultiScale(img, faces);
                
                //Loop to draw rectangles on all detected faces
                for(size_t j=0; j<faces.size(); j++){
                    Point pt1(faces[j].x, faces[j].y);
                    Point pt2((faces[j].x + faces[j].height), (faces[j].y + faces[j].width));
                    rectangle(img,pt1,pt2,Scalar(0, 0, 255), 2, 8, 0);
                }
                
                //name of output image
                string out = "Output/out_"; 
                out += img_names.at(i);

                //Save image
                imwrite(out, img);

                cout<<out+"\n";
            }
        }
    }
    //end time
    t2 = omp_get_wtime();

    //Total run time
    run_time = t2 - t1;
    
    cout<<"parallel face detection time = "<<run_time<<"\n";
}
