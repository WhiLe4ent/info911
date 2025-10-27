#include <cstdio>
#include <iostream>
#include <algorithm>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "ColorDistribution.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
  Mat img_input;
  VideoCapture *pCap = nullptr;
  const int width = 640;
  const int height = 480;
  const int size = 50;
  const int bbloc = 128;
  const float DIST_THRESHOLD = 0.05f;

  // Ouvre la camera
  pCap = new VideoCapture(0);
  if (!pCap->isOpened())
  {
    cout << "Couldn't open image / camera ";
    return 1;
  }

  // Force une camera 640x480
  pCap->set(CAP_PROP_FRAME_WIDTH, width);
  pCap->set(CAP_PROP_FRAME_HEIGHT, height);

  (*pCap) >> img_input;
  if (img_input.empty())
    return 1; // problème avec la caméra

  Point pt1(width / 2 - size / 2, height / 2 - size / 2);
  Point pt2(width / 2 + size / 2, height / 2 + size / 2);

  namedWindow("input", 1);
  imshow("input", img_input);

  bool freeze = false;
  bool reco = false;

  vector<vector<ColorDistribution>> all_col_hists;
  all_col_hists.push_back(vector<ColorDistribution>());

  vector<Vec3b> colors = {
      Vec3b(0, 0, 0),
      Vec3b(0, 0, 255),
      Vec3b(0, 255, 0),
      Vec3b(255, 0, 0),
      Vec3b(0, 255, 255),
      Vec3b(255, 0, 255),
      Vec3b(255, 255, 255)
  };

  int current_object = -1;

  cout << "=== Commandes disponibles ===" << endl;
  cout << " b : apprendre le fond" << endl;
  cout << " n : créer un nouvel objet" << endl;
  cout << " a : ajouter un échantillon à l'objet courant" << endl;
  cout << " r : activer/désactiver la reconnaissance" << endl;
  cout << " v : comparer gauche/droite (test distance)" << endl;
  cout << " f : geler/dégeler la caméra" << endl;
  cout << " q / ESC : quitter" << endl;
  cout << "=============================" << endl;

  while (true)
  {
    char c = (char)waitKey(50); // attend 50ms -> 20 images/s

    if (pCap != nullptr && !freeze)
      (*pCap) >> img_input; // récupère l'image de la caméra

    if (c == 27 || c == 'q') // permet de quitter l'application
      break;

    if (c == 'f')
      freeze = !freeze;

    if (c == 'b')
    {
      all_col_hists[0].clear();
      for (int y = 0; y <= height - bbloc; y += bbloc)
        for (int x = 0; x <= width - bbloc; x += bbloc)
        {
          ColorDistribution cd = getColorDistribution(img_input, Point(x, y), Point(x + bbloc, y + bbloc));
          addDistributionIfFar(all_col_hists[0], cd, DIST_THRESHOLD);
        }
      cout << "Fond enregistré (" << all_col_hists[0].size() << " distributions uniques)." << endl;
    }

    if (c == 'n')
    {
      all_col_hists.push_back(vector<ColorDistribution>());
      current_object = (int)all_col_hists.size() - 1;
      cout << "Nouvel objet créé (index " << current_object << ")." << endl;
    }

    if (c == 'a' && current_object >= 1)
    {
      ColorDistribution cd = getColorDistribution(img_input, pt1, pt2);
      addDistributionIfFar(all_col_hists[current_object], cd, DIST_THRESHOLD);
      cout << "Échantillons ajouté à l'objet " << current_object
           << " (" << all_col_hists[current_object].size() << " distributions uniques)." << endl;
    }

    if (c == 'r')
      reco = !reco;

    if (c == 'v')
    {
      ColorDistribution left = getColorDistribution(img_input, Point(0, 0), Point(width / 2, height));
      ColorDistribution right = getColorDistribution(img_input, Point(width / 2, 0), Point(width, height));
      float d = left.distance(right);
      cout << "Distance gauche/droite = " << d << endl;
    }

    Mat output = img_input;

    if (reco && all_col_hists.size() > 1 && !all_col_hists[0].empty())
    {
      Mat gray;
      cvtColor(img_input, gray, COLOR_BGR2GRAY);
      Mat reco_img = recoObjectMulti(img_input, all_col_hists, colors, 8);
      cvtColor(gray, img_input, COLOR_GRAY2BGR);
      output = 0.5 * reco_img + 0.5 * img_input;
    }
    else
    {
      cv::rectangle(img_input, pt1, pt2, Scalar({255.0, 255.0, 255.0}), 1);
    }

    imshow("input", output);
  }

  return 0;
}
