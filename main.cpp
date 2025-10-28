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

static void putOverlay(Mat &img, const vector<string> &lines)
{
  const int margin = 8;
  const double scale = 0.5;
  int y = margin + 12;
  for (const auto &line : lines)
  {
    putText(img, line, Point(margin, y), FONT_HERSHEY_SIMPLEX, scale, Scalar(200, 200, 200), 1, LINE_AA);
    y += 16;
  }
}

int main(int argc, char **argv)
{
  Mat img_input;
  VideoCapture pCap(0);
  const int width = 640;
  const int height = 480;
  const int size = 50;
  const int bbloc = 128;
  float DIST_THRESHOLD = 0.005f;
  int small_bloc = 8;
  bool useRelaxDefault = true;
  int superFactorDefault = 2;

  // Ouvre la camera
  if (!pCap.isOpened())
  {
    cout << "Couldn't open image / camera ";
    return 1;
  }

  // Force une camera 640x480
  pCap.set(CAP_PROP_FRAME_WIDTH, width);
  pCap.set(CAP_PROP_FRAME_HEIGHT, height);
  pCap >> img_input;
  if (img_input.empty())
    return 1; // problème avec la caméra

  Point pt1(width / 2 - size / 2, height / 2 - size / 2);
  Point pt2(width / 2 + size / 2, height / 2 + size / 2);

  namedWindow("input", 1);

  bool freeze = false;
  bool reco = false;
  bool show_relaxed = useRelaxDefault;

  vector<vector<ColorDistribution>> all_col_hists(1);

  vector<Vec3b> colors = {
      Vec3b(0, 0, 0),
      Vec3b(0, 0, 255),
      Vec3b(0, 255, 0),
      Vec3b(255, 0, 0),
      Vec3b(0, 255, 255),
      Vec3b(255, 0, 255),
      Vec3b(255, 255, 255)};

  int current_object = -1;

  cout << "\n=== Commandes disponibles ===" << endl;
  cout << " b : apprendre le fond" << endl;
  cout << " n : créer un nouvel objet" << endl;
  cout << " a : ajouter un échantillon à l'objet courant" << endl;
  cout << " r : activer/désactiver la reconnaissance" << endl;
  cout << " v : comparer gauche/droite (test distance)" << endl;
  cout << " f : geler/dégeler la caméra" << endl;
  cout << " g : activer/désactiver le lissage (relaxation)" << endl;
  cout << " +/- : augmenter/diminuer DIST_THRESHOLD (filtre doublons)" << endl;
  cout << " s/S : augmenter/diminuer superFactor (grouping)" << endl;
  cout << " q / ESC : quitter" << endl;
  cout << "=============================\n"
       << endl;

  while (true)
  {
    char c = (char)waitKey(50); // attend 50ms -> 20 images/s

    if (!freeze)
    {
      pCap >> img_input;
      if (img_input.empty())
        continue;
    }

    if (c == 27 || c == 'q')
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
      cout << "Fond appris (" << all_col_hists[0].size() << " distributions uniques)." << endl;
    }

    if (c == 'n')
    {
      all_col_hists.push_back(vector<ColorDistribution>());
      current_object = (int)all_col_hists.size() - 1;
      cout << "Nouvel objet créé : index " << current_object << endl;
    }

    if (c == 'a')
    {
      if (current_object < 1)
      {
        cout << "Erreur : crée d'abord un objet avec 'n' avant d'ajouter des échantillons." << endl;
      }
      else
      {
        ColorDistribution cd = getColorDistribution(img_input, pt1, pt2);
        addDistributionIfFar(all_col_hists[current_object], cd, DIST_THRESHOLD);
        cout << "Échantillon ajouté à l'objet " << current_object
             << " (" << all_col_hists[current_object].size() << " distributions uniques)." << endl;
      }
    }

    if (c == 'r')
    {
      reco = !reco;
      cout << "Reconnaissance : " << (reco ? "ON" : "OFF") << endl;
    }

    if (c == 'g')
    {
      show_relaxed = !show_relaxed;
      cout << "Mode lissage : " << (show_relaxed ? "activé" : "désactivé") << endl;
    }

    if (c == '+' || c == '=')
    {
      DIST_THRESHOLD = std::min(1.f, DIST_THRESHOLD + 0.005f);
      cout << "DIST_THRESHOLD = " << DIST_THRESHOLD << endl;
    }
    if (c == '-' || c == '_')
    {
      DIST_THRESHOLD = std::max(0.f, DIST_THRESHOLD - 0.005f);
      cout << "DIST_THRESHOLD = " << DIST_THRESHOLD << endl;
    }

    if (c == 's')
    {
      superFactorDefault = std::min(8, superFactorDefault + 1);
      cout << "superFactor = " << superFactorDefault << endl;
    }
    if (c == 'S')
    {
      superFactorDefault = std::max(1, superFactorDefault - 1);
      cout << "superFactor = " << superFactorDefault << endl;
    }

    if (c == 'v')
    {
      ColorDistribution left = getColorDistribution(img_input, Point(0, 0), Point(width / 2, height));
      ColorDistribution right = getColorDistribution(img_input, Point(width / 2, 0), Point(width, height));
      cout << "Distance gauche/droite = " << left.distance(right) << endl;
    }

    Mat output = img_input.clone();

    if (reco && all_col_hists.size() > 1 && !all_col_hists[0].empty())
    {
      if (current_object < 1 && all_col_hists.size() > 1)
        current_object = 1;

      int sf = show_relaxed ? superFactorDefault : 1;

      std::vector<std::vector<int>> labels;
      Mat reco_img = recoObjectMulti(img_input, all_col_hists, colors, small_bloc, labels, show_relaxed, sf);

      Mat markers = computeMarkers(labels, small_bloc, sf);

      Mat img_for_ws;
      img_input.copyTo(img_for_ws);
      cv::watershed(img_for_ws, markers);

      Mat final = Mat::zeros(img_input.size(), CV_8UC3);
      for (int y = 0; y < markers.rows; ++y)
      {
        for (int x = 0; x < markers.cols; ++x)
        {
          int idx = markers.at<int>(y, x);
          if (idx > 0 && idx < (int)colors.size())
            final.at<Vec3b>(y, x) = colors[idx];
        }
      }

      addWeighted(final, 0.7, img_input, 0.3, 0.0, output);
    }
    else
    {
      rectangle(output, pt1, pt2, Scalar(255, 255, 255), 1);
    }

    vector<string> lines;
    lines.push_back("b:fond  n:newObj  a:addSample  r:reco  g:relax  +/-:thresh  s/S:superFactor");
    lines.push_back(string("Recon:") + (reco ? "ON " : "OFF ") +
                    "  Lissage:" + (show_relaxed ? "ON " : "OFF ") +
                    "  Thresh:" + to_string(DIST_THRESHOLD));
    lines.push_back(string("NbObjs:") + to_string((int)all_col_hists.size() - 1) +
                    "  Current:" + to_string(current_object));
    putOverlay(output, lines);

    imshow("input", output);
  }

  return 0;
}
