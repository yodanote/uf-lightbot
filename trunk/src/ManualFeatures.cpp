#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include<fstream>
#include <string>
#include <ctype.h>


using namespace cv;
using namespace std;

Point2f pt;
bool addRemovePt = false;
bool done = false;
Mat img;



void finddot(char *in_str, char *out_str) 
{ int i;
	while(&in_str[i] != ".") 
	{out_str[i] = in_str[i]; i ++ ;}
	out_str[i] = '\0'; 
}

void onMouse( int event, int x, int y, int flags, void* param )
{
    if( event == CV_EVENT_LBUTTONDOWN )
    {
        pt = Point2f((float)x,(float)y);
       	addRemovePt = true;
    }
    
}

int main( int argc, char** argv )
{

   int count = 0;
   const int MAX_COUNT = 15;
   ofstream outfile;
   string filename = argc >=2 ? argv[1] : "lena.jpg";
   string outfilename;
   int loc = filename.find(".");
   outfilename = filename.substr(0, loc);
   outfilename = outfilename + "out.txt";
   cout << outfilename << endl;


    Mat img = imread(filename.c_str());
 
    namedWindow( "LK Demo", 1 );
    setMouseCallback( "LK Demo", onMouse);

    vector<Point2f> points;
     imshow ("LK Demo", img);
     cout << "Place nose location" << endl;
      while(1){


      if( addRemovePt && points.size() < 15)
      {
         points.push_back(pt);
	 if (count == 0){
		  circle(img, pt, 3, Scalar(0, 255, 0), -1, 8);
		 cout << "Place Left mouth location" << endl;
	 }
      	 else if (count == 1){
		  circle(img, pt, 3, Scalar(0, 255, 0), -1, 8);
		 cout << "Place Top mouth location" << endl;
	 }
      	 else if (count == 2){
		  circle(img, pt, 3, Scalar(0, 255, 0), -1, 8);
		 cout << "Place Right mouth location" << endl;
	 }
      	 else if (count == 3){
		  circle(img, pt, 3, Scalar(0, 255, 0), -1, 8);
		 cout << "Place Bottom mouth location" << endl;
	 }

       	 else if (count == 4){
		  circle(img, pt, 3, Scalar(0, 255, 0), -1, 8);
		 cout << "Place top left eye location" << endl;
	 }
	else if (count == 5){
	 	circle( img, pt, 3, Scalar(255,0,0), -1, 8);   
	        cout<<"Place bottom left eye location"<<endl;
        }

	else if (count == 6){
	 	circle( img, pt, 3, Scalar(255,0,0), -1, 8);   
	        cout<<"Place  top right eye location"<<endl;
        }
	else if (count == 7){
	 	circle( img, pt, 3, Scalar(255,0,0), -1, 8);   
	        cout<<"Place bottom right  eye location"<<endl;
        }

	else if (count == 8){
	 	circle( img, pt, 3, Scalar(255,0,0), -1, 8);   
	        cout<<"Place left, left brow location"<<endl;
        }
	else if (count == 9){
		circle(img, pt, 3, Scalar(0, 0, 255), -1, 8);
                cout<<"Place middle, left brow location"<<endl;
  	}
        else if (count == 10){
		circle(img, pt, 3, Scalar(0, 0, 255), -1, 8);
                cout<<"Place right, left brow location"<<endl;
  	}

	else if (count == 11){
	 	circle( img, pt, 3, Scalar(255,0,0), -1, 8);   
	        cout<<"Place left, right brow location"<<endl;
        }
	else if (count == 12){
		circle(img, pt, 3, Scalar(0, 0, 255), -1, 8);
                cout<<"Place middle, right brow location"<<endl;
  	}
        else if (count == 13){
		circle(img, pt, 3, Scalar(0, 0, 255), -1, 8);
                cout<<"Place right, right brow location"<<endl;
  	}
        else {
    	  if (count==14){
	   done = true;
	   cout << "Done!" << endl;
	  }
	}
         addRemovePt = false;
	 imshow("LK Demo", img);
	 count++;

	if (done == true){
		        outfile.open(outfilename.c_str());
			outfile<< "nose: x=" << points[0].x << "  y=" << points[0].y << endl;
			
                        outfile<< "mouth_left: x=" << points[1].x << "  y=" << points[1].y << endl;
			outfile<< "mouth_top: x=" << points[2].x << "  y=" << points[2].y << endl;
			outfile<< "mouth_right: x=" << points[3].x << "  y=" << points[3].y << endl;
			outfile<< "mouth_bottom: x=" << points[4].x << "  y=" << points[4].y << endl;

			outfile<< "lefteye_top: x=" << points[5].x << "  y=" << points[5].y << endl;
			outfile<< "lefteye_bottom: x=" << points[6].x << "  y=" << points[6].y << endl;		
			
                        outfile<< "righteye_top: x=" << points[7].x << "  y=" << points[7].y << endl;
			outfile<< "righteye_bottom: x=" << points[8].x << "  y=" << points[8].y << endl;
			
                        outfile<< "leftbrow_left: x=" << points[9].x << "  y=" << points[9].y << endl;
			outfile<< "leftbrow_middle: x=" << points[10].x << "  y=" << points[10].y << endl;
			outfile<< "leftbrow_right: x=" << points[11].x << "  y=" << points[11].y << endl;
			
                        outfile<< "rightbrow_left: x=" << points[12].x << "  y=" << points[12].y << endl;
			outfile<< "rightbrow_middle: x=" << points[13].x << "  y=" << points[13].y << endl;
			outfile<< "rightbrow_right: x=" << points[14].x << "  y=" << points[14].y << endl;
		        
                        outfile.close();
		      done= false;
		     break;
		
		}

       }
       if( cvWaitKey( 15 )==27 ) 
			break;
    }
    
}

