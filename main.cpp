#include <iostream>
#include <opencv2/opencv.hpp>
#include <limits.h>
#include <chrono>

#include "ORBextractor.h"

#define MAX_DELTAY 0
#define MAX_DELTAX 32

using  namespace cv;

cv::Mat drawMatches_(const cv::Mat &left_image, const cv::Mat &right_image,
                                 const std::vector<Point2f> &kpts_l, std::vector<Point2f> &kpts_r,
                                 const std::vector<cv::DMatch> &matches) {

    cv::Mat imageMatches;
    //convert vector of Point2f to vector of Keypoint
    std::vector<KeyPoint> prevPoints, nextPoints;
    for (int i = 0; i < kpts_l.size(); i++){
        KeyPoint kpt_l, kpt_r;
        kpt_l.pt = kpts_l.at(i);
        kpt_r.pt = kpts_r.at(i);
        prevPoints.push_back(kpt_l);
        nextPoints.push_back(kpt_r);
    }

    drawMatches(left_image, prevPoints, right_image, nextPoints, matches, imageMatches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::DEFAULT);

    return imageMatches;

}

cv::Mat drawKeypoints_ (const cv::Mat &im, const std::vector<Point2f> &pts){

    cv::Mat imageKpts;
    //convert vector of Point2f to vector of Keypoint
    std::vector<KeyPoint> kpts;
    for (int i = 0; i < pts.size(); i++){
        KeyPoint kpt;
        kpt.pt = pts.at(i);
        kpts.push_back(kpt);
    }

    drawKeypoints( im, kpts, imageKpts, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    return imageKpts;

}

double euclideanDist(const cv::Point2d &p, const cv::Point2d &q) {
    Point2d diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

bool findMatchingSAD(const cv::Point2f &pt_l, const cv::Mat& imLeft, const cv::Mat& imRight,
                     std::vector<Point2f>& pts_r, Point2f &ptr_m, int &index){

    int halfBlockSize = 3;
    int blockSize = 2 * halfBlockSize + 1;

    int width = imRight.size().width;
    int height = imRight.size().height;

    Mat template_(blockSize, blockSize, CV_64F);
    //get pixel neighbors
    //        Mat template_ = imLeft(Rect ((int)pt.x - halfBlockSize, (int)pt.y - halfBlockSize, halfBlockSize, halfBlockSize)).clone();
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < blockSize; j++) {
            int x = (int) pt_l.x - (halfBlockSize - i);
            int y = (int) pt_l.y - (halfBlockSize - j);
            //check frame limits
            if (x >= 0 && x < width && y >= 0 && y < height) {
                Scalar intensity = imLeft.at<uchar>(y, x);
                template_.at<float>(j, i) = (int) intensity[0];
            } else {
                template_.at<float>(j, i) = 0;
            }
        }
    }

    int minSAD = 1000;
    Point2f bestPt;

    //flag to know when the point has no matching
    bool noMatching = true;

    int index_r = 0, bestIndex_r = 0;
    for (auto &pt_r:pts_r) {

        if(!(pt_r.x == -1 && pt_r.y == -1)){

            int deltay = (int) abs(pt_l.y - pt_r.y);
            int deltax = (int) pt_l.x - (int) pt_r.x;

            //epipolar constraints, the correspondent keypoint must be in the same row and disparity should be positive
            if (deltax >= 0 && deltay <= MAX_DELTAY && abs (deltax) <= MAX_DELTAX) {


                //compute SAD
                int sum = 0;
                for (int i = 0; i < blockSize; i++) {
                    for (int j = 0; j < blockSize; j++) {
                        int x = (int) pt_r.x - (halfBlockSize - i);
                        int y = (int) pt_r.y - (halfBlockSize - j);
                        //check frame limits
                        if (x >= 0 && x < width && y >= 0 && y < height) {
                            Scalar intensity = imRight.at<uchar>(y, x);
                            sum += abs(template_.at<float>(j, i) - intensity[0]);
                        } else {
                            sum += abs(template_.at<float>(j, i) - 0);
                        }
                    }
                }

                if (sum < minSAD) {
                    noMatching = false;
                    minSAD = sum;
                    bestPt = pt_r;
                    bestIndex_r = index_r;
                }
            }

        }else {
//            std::cout << "Point have been matched: " << std::endl;
        }
//        std::cout << "index_r: " << index_r << std::endl;
        index_r ++;
    }



    if (!noMatching) {
//        std::cout << "Min SAD: " << minSAD << std::endl;
//        std::cout << "bestIndex_r: " << bestIndex_r << std::endl;

        std::cout << "Disparity: " << pt_l.x - bestPt.x << "\n";
        pts_r[bestIndex_r] = Point (-1,-1);
        ptr_m = bestPt;
        index = bestIndex_r;

        return true;
    }else{
        return false;
    }

}


void stereoMatching_(std::vector<Point2f>& pts_l, std::vector<Point2f>& pts_r, const cv::Mat& imLeft,
                    const cv::Mat& imRight, const std::vector<bool>& inliers,  std::vector<cv::DMatch> &matches){

//    TermCriteria criteria = TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );
//
//    cornerSubPix(imLeft, pts_l, Size(7,7), Size(-1,-1), criteria);

    // The disparity range defines how many pixels away from the block's location
    // in the first image to search for a matching block in the other image.
    int disparityRange = 50;

//    imshow("Image Right", imRight);
//
//    waitKey(0);

    //Define the size of the blocks for block matching.
    int halfBlockSize = 3;
    int blockSize = 2 * halfBlockSize + 1;

    int width = imRight.size().width;
    int height = imRight.size().height;

    int pos = 0;
    for (auto &pt:pts_l){
        Mat template_ (blockSize, blockSize, CV_64F);
        //get pixel neighbors
        //        Mat template_ = imLeft(Rect ((int)pt.x - halfBlockSize, (int)pt.y - halfBlockSize, halfBlockSize, halfBlockSize)).clone();
        for (int i=0; i < blockSize; i++){
            for(int j=0; j < blockSize; j++){
                int x = (int) pt.x - (halfBlockSize - i);
                int y = (int) pt.y - (halfBlockSize - j);
                //check frame limits
                if(x >= 0 && x < width && y >= 0 && y < height){
                    Scalar intensity = imLeft.at<uchar>(y,x);
                    template_.at<float>(j,i) = (int) intensity[0];
                }else{
                    template_.at<float>(j,i) = 0;
                }
            }
        }

        // Set the min column bounds for the template search.
        int minc = MAX(blockSize, (int) pt.x - disparityRange);

        int minSAD = INT_MAX;
        Point2f bestPt;


        for (int k = pt.x; k >= minc; k--){

            Point point (k, pt.y);

//            Mat block = imRight(Rect (i - halfBlockSize, i - halfBlockSize, halfBlockSize, halfBlockSize)).clone();

            //compute SAD
            int sum =0;
            for (int i=0; i < blockSize; i++){
                for(int j=0; j < blockSize; j++){
                    int x = (int) point.x - (halfBlockSize - i);
                    int y = (int) point.y - (halfBlockSize - j);
                    //check frame limits
                    if(x >= 0 && x < width && y >= 0 && y < height){
                        Scalar intensity = imRight.at<uchar>(y,x);
                        sum += abs(template_.at<float>(j,i) - intensity[0]);
                    }else{
                        sum += abs(template_.at<float>(j,i) - 0);
                    }
                }
            }

            if(sum < minSAD){
                minSAD = sum;
                bestPt = point;
            }

        }
        pts_r.push_back(bestPt);
        double dst = norm(Mat(pts_l.at(pos)), Mat(pts_r.at(pos)));
        DMatch match (pos,pos, dst);
        matches.push_back(match);

        pos ++;
    }

//    cornerSubPix(imRight, pts_r, Size(7,7), Size(-1,-1), criteria);

}


void stereoMatching(const std::vector<Point2f>& pts_l,  const std::vector<Point2f>& pts_r, const cv::Mat& imLeft,
                    const cv::Mat& imRight, std::vector<cv::DMatch> &matches,
                    std::vector<Point2f> &new_pts_l, std::vector<Point2f> &new_pts_r) {

    std::vector<Point2f> aux_pts_r(pts_r);

    int pos = 0;
    int index_l = 0;
    for (auto &pt_l:pts_l) {

        Point2f ptr;
        int index;
        bool found = findMatchingSAD(pt_l, imLeft, imRight, aux_pts_r, ptr, index);
        if(found){
            new_pts_l.push_back(pt_l);
            new_pts_r.push_back(ptr);

            double dst = euclideanDist(pt_l, ptr);
            DMatch match(pos, pos, dst);
            matches.push_back(match);
            pos++;

        }

        index_l ++;
    }


    std::cout << "remaining left points: " << new_pts_l.size() << std::endl;
    std::cout << "remaining right points: "<< new_pts_r.size() << std::endl;
    std::cout << "Number of matches: "<< matches.size()   << std::endl;

}




int main() {

    //loading images
    std::string path = "/media/nigel/Dados/Documents/Projetos/CLionProjects/stereoMatching/images/";
    Mat left_frame  = imread(path + "left_0.png");
    Mat right_frame = imread(path + "right_0.png");


    if(left_frame.channels() == 3)
        cvtColor(left_frame, left_frame, cv::COLOR_RGB2GRAY);
    if(right_frame.channels() == 3)
        cvtColor(right_frame, right_frame, cv::COLOR_RGB2GRAY);

//    Ptr<FeatureDetector> detector = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::FAST_SCORE, 31);

    ORBextractor detector(500, 1.2, 8, 20, 7);

    std::vector<KeyPoint> kpts_l, kpts_r;
//    detector -> detect(left_frame, kpts_l);
//    detector -> detect(right_frame, kpts_r);

    detector(left_frame, cv::Mat(), kpts_l);
    detector(right_frame, cv::Mat(), kpts_r);

//    Mat im_kpts;
//    drawKeypoints( left_frame, kpts_l, im_kpts, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

    std::cout << "Num points detected left: " << kpts_l.size() << std::endl;
    std::cout << "Num points detected right: "<< kpts_r.size() << std::endl;

    //convert vector of keypoints to vector of Point2f
    std::vector<Point2f> prevPoints, nextPoints, pts_l, pts_r;
    for (auto& kpt:kpts_l)
        prevPoints.push_back(kpt.pt);

    for (auto &kpt:kpts_r)
        nextPoints.push_back(kpt.pt);


    std::vector<bool> inliers;
    std::vector<cv::DMatch> matches;

    auto startTime = std::chrono::steady_clock::now();

//    stereoMatching_(prevPoints, nextPoints, left_frame, right_frame, inliers, matches);
    stereoMatching (prevPoints, nextPoints, left_frame, right_frame, matches, pts_l, pts_r);

    auto endTime   = std::chrono::steady_clock::now();

    std::cout << "Elapsed time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()
              << " ms" << std::endl;

//    Mat image_mathes = drawMatches_(left_frame, right_frame, prevPoints, nextPoints, matches);
    Mat image_mathes = drawMatches_(left_frame, right_frame, pts_l, pts_r, matches);

    Mat imKpts_l = drawKeypoints_(left_frame, pts_l);
    Mat imKpts_r = drawKeypoints_(right_frame, pts_r);

//    imshow("Kpts left", im_kpts);
    imshow("Kpts left remaining", imKpts_l);
    imshow("Kpts right remaining", imKpts_r);
    imshow("matches", image_mathes);
    waitKey(0);



    return 0;
}