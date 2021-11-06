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


int main() {
    double run_time = 0;
    const char path[] = "Input";
    struct dirent* entry;
    DIR* dir = opendir(path);

    if (dir == NULL) {
        cout << "input directory missing\n";
        return 0;
    }

    double t1, t2;
    vector<string> img_names;
    while ((entry = readdir(dir)) != NULL) {
        img_names.push_back(entry->d_name);
    }

    omp_set_num_threads(8);

    int N = img_names.size();
    int tid, numt, i;

    t1 = omp_get_wtime();

    #pragma omp parallel default(shared) private(tid, numt, i)
    {
        CascadeClassifier FD;

        if (!FD.load("haarcascade_frontalface_default.xml")) {
            cout << "No load file\n";
            exit(0);
        }

        int from, to;

        tid = omp_get_thread_num();
        numt = omp_get_num_threads();

        from = (N / numt) * tid;
        to = (N / numt) * (tid + 1) - 1;

        if (tid == numt - 1) {
            to = N - 1;
        }

        for (i = from; i <= to; i++) {
            string img_path = path;
            img_path += "/";
            img_path += img_names.at(i);

            Mat img = imread(img_path, IMREAD_UNCHANGED);

            if (img.empty()) {
                continue;
            }
            else {
                vector<Rect> faces;

                // #pragma omp critical
                FD.detectMultiScale(img, faces);

                for (size_t j = 0; j < faces.size(); j++) {
                    Point pt1(faces[j].x, faces[j].y);
                    Point pt2((faces[j].x + faces[j].height), (faces[j].y + faces[j].width));
                    rectangle(img, pt1, pt2, Scalar(0, 0, 255), 2, 8, 0);
                }

                string out = "Output/out_";
                out += img_names.at(i);
                imwrite(out, img);

                cout << out + "\n";
            }
        }
    }
    t2 = omp_get_wtime();
    run_time = t2 - t1;
    cout << "parallel face detection time = " << run_time << "\n";
}
