#include "ColorDistribution.hpp"
#include <cfloat>
#include <algorithm>
#include <map>

void ColorDistribution::reset()
{
    nb = 0;
    for (int r = 0; r < 8; r++)
        for (int g = 0; g < 8; g++)
            for (int b = 0; b < 8; b++)
                data[r][g][b] = 0.f;
}

void ColorDistribution::add(Vec3b color)
{
    int r = color[2] / 32;
    int g = color[1] / 32;
    int b = color[0] / 32;

    r = std::min(7, std::max(0, r));
    g = std::min(7, std::max(0, g));
    b = std::min(7, std::max(0, b));
    data[r][g][b] += 1.f;
    nb++;
}

void ColorDistribution::finished()
{
    if (nb == 0)
        return;
    for (int r = 0; r < 8; r++)
        for (int g = 0; g < 8; g++)
            for (int b = 0; b < 8; b++)
                data[r][g][b] /= static_cast<float>(nb);
}

float ColorDistribution::distance(const ColorDistribution &other) const
{
    float d = 0.f;
    for (int r = 0; r < 8; r++)
        for (int g = 0; g < 8; g++)
            for (int b = 0; b < 8; b++)
            {
                float a = data[r][g][b];
                float b2 = other.data[r][g][b];
                float denom = a + b2;
                if (denom > 0.f)
                    d += (a - b2) * (a - b2) / denom;
            }
    return d * 0.5f;
}

ColorDistribution getColorDistribution(const Mat &input, Point pt1, Point pt2)
{
    ColorDistribution cd;

    int x1 = std::max(0, std::min(pt1.x, input.cols - 1));
    int y1 = std::max(0, std::min(pt1.y, input.rows - 1));
    int x2 = std::max(0, std::min(pt2.x, input.cols));
    int y2 = std::max(0, std::min(pt2.y, input.rows));
    if (x2 <= x1 || y2 <= y1)
        return cd;

    for (int y = y1; y < y2; y++)
        for (int x = x1; x < x2; x++)
            cd.add(input.at<Vec3b>(y, x));
    cd.finished();
    return cd;
}

float minDistance(const ColorDistribution &h, const std::vector<ColorDistribution> &hists)
{
    if (hists.empty())
        return FLT_MAX;
    float dmin = FLT_MAX;
    for (const auto &hh : hists)
    {
        float d = h.distance(hh);
        if (d < dmin)
        {
            dmin = d;
            if (dmin <= 0.f)
                return 0.f;
        }
    }
    return dmin;
}

void addDistributionIfFar(std::vector<ColorDistribution> &hists,
                          const ColorDistribution &newHist,
                          float threshold)
{
    if (newHist.nb == 0)
        return;
    if (hists.empty())
    {
        hists.push_back(newHist);
        return;
    }
    float d = minDistance(newHist, hists);
    if (d > threshold)
    {
        hists.push_back(newHist);
    }
}

static inline int majorityLabelAt(const std::vector<std::vector<int>> &labels, int r, int c, int nbRows, int nbCols)
{
    std::map<int, int> counts;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int nr = r + dy;
            int nc = c + dx;
            if (nr >= 0 && nr < nbRows && nc >= 0 && nc < nbCols)
            {
                counts[labels[nr][nc]]++;
            }
        }
    }
    int best = labels[r][c];
    int bestCount = 0;
    for (const auto &p : counts)
    {
        if (p.second > bestCount)
        {
            best = p.first;
            bestCount = p.second;
        }
    }
    return best;
}

void relaxLabels(std::vector<std::vector<int>> &labels, int nbRows, int nbCols, int passes)
{
    if (nbRows <= 0 || nbCols <= 0)
        return;
    std::vector<std::vector<int>> tmp = labels;
    for (int pass = 0; pass < passes; ++pass)
    {
        for (int r = 0; r < nbRows; ++r)
        {
            for (int c = 0; c < nbCols; ++c)
            {
                tmp[r][c] = majorityLabelAt(labels, r, c, nbRows, nbCols);
            }
        }
        labels.swap(tmp);
    }
}

cv::Mat recoObject(const cv::Mat &input,
                   const std::vector<ColorDistribution> &col_hists,
                   const std::vector<ColorDistribution> &col_hists_object,
                   const std::vector<cv::Vec3b> &colors,
                   const int bloc)
{
    Mat output = Mat::zeros(input.size(), CV_8UC3);

    for (int y = 0; y < input.rows; y += bloc)
    {
        for (int x = 0; x < input.cols; x += bloc)
        {
            Point p1(x, y);
            Point p2(std::min(x + bloc, input.cols), std::min(y + bloc, input.rows));
            ColorDistribution h = getColorDistribution(input, p1, p2);

            float d_fond = minDistance(h, col_hists);
            float d_obj = minDistance(h, col_hists_object);

            int label = (d_obj < d_fond) ? 1 : 0;
            Vec3b col = (label >= 0 && label < (int)colors.size()) ? colors[label] : Vec3b(0, 0, 0);
            rectangle(output, p1, p2, Scalar(col), FILLED);
        }
    }

    return output;
}

int closestObjectIndex(const ColorDistribution &h,
                       const std::vector<std::vector<ColorDistribution>> &all_hists)
{
    int best_index = -1;
    float best_dist = FLT_MAX;
    for (size_t i = 0; i < all_hists.size(); ++i)
    {
        if (all_hists[i].empty())
            continue;
        float d = minDistance(h, all_hists[i]);
        if (d < best_dist)
        {
            best_dist = d;
            best_index = static_cast<int>(i);
            if (best_dist <= 0.f)
                break;
        }
    }
    if (best_index < 0)
    {
        best_index = 0;
    }
    return best_index;
}

cv::Mat recoObjectMulti(const cv::Mat &input,
                        const std::vector<std::vector<ColorDistribution>> &all_col_hists,
                        const std::vector<cv::Vec3b> &colors,
                        int bloc,
                        std::vector<std::vector<int>> &outLabels,
                        bool doRelax,
                        int superFactor)
{
    const int rowsBlocs = (input.rows + bloc - 1) / bloc;
    const int colsBlocs = (input.cols + bloc - 1) / bloc;

    std::vector<std::vector<int>> labels(rowsBlocs, std::vector<int>(colsBlocs, 0));
    for (int by = 0; by < rowsBlocs; ++by)
    {
        for (int bx = 0; bx < colsBlocs; ++bx)
        {
            int x = bx * bloc;
            int y = by * bloc;
            Point p1(x, y);
            Point p2(std::min(x + bloc, input.cols), std::min(y + bloc, input.rows));
            ColorDistribution cd = getColorDistribution(input, p1, p2);
            int obj_idx = closestObjectIndex(cd, all_col_hists);
            labels[by][bx] = obj_idx;
        }
    }

    if (doRelax)
        relaxLabels(labels, rowsBlocs, colsBlocs, 3);

    if (superFactor < 1)
        superFactor = 1;
    int sRows = (rowsBlocs + superFactor - 1) / superFactor;
    int sCols = (colsBlocs + superFactor - 1) / superFactor;

    Mat reco(input.size(), CV_8UC3, Scalar(0, 0, 0));

    for (int sy = 0; sy < sRows; ++sy)
    {
        for (int sx = 0; sx < sCols; ++sx)
        {
            std::map<int, int> counts;
            for (int dy = 0; dy < superFactor; ++dy)
            {
                for (int dx = 0; dx < superFactor; ++dx)
                {
                    int by = sy * superFactor + dy;
                    int bx = sx * superFactor + dx;
                    if (by >= rowsBlocs || bx >= colsBlocs)
                        continue;
                    counts[labels[by][bx]]++;
                }
            }
            int bestLabel = 0;
            int bestCount = 0;
            for (const auto &p : counts)
            {
                if (p.second > bestCount)
                {
                    bestLabel = p.first;
                    bestCount = p.second;
                }
            }
            int x1 = sx * superFactor * bloc;
            int y1 = sy * superFactor * bloc;
            int x2 = std::min(input.cols, (sx + 1) * superFactor * bloc);
            int y2 = std::min(input.rows, (sy + 1) * superFactor * bloc);
            int colorIdx = bestLabel % std::max(1, (int)colors.size());
            Vec3b color = (bestLabel >= 0 && bestLabel < (int)colors.size()) ? colors[colorIdx] : colors[colorIdx];
            rectangle(reco, Point(x1, y1), Point(x2 - 1, y2 - 1), Scalar(color), FILLED);
        }
    }

    for (int by = 1; by < rowsBlocs - 1; ++by)
    {
        for (int bx = 1; bx < colsBlocs - 1; ++bx)
        {
            int current = labels[by][bx];
            if (labels[by - 1][bx] != current ||
                labels[by + 1][bx] != current ||
                labels[by][bx - 1] != current ||
                labels[by][bx + 1] != current)
            {
                int x = bx * bloc;
                int y = by * bloc;
                rectangle(reco, Point(x, y), Point(x + bloc - 1, y + bloc - 1), Scalar(0, 0, 0), 1);
            }
        }
    }

    outLabels = labels;
    return reco;
}

cv::Mat computeMarkers(const std::vector<std::vector<int>> &labels, int bloc, int superFactor)
{
    const int rows = labels.size();
    const int cols = labels[0].size();

    cv::Mat markers = cv::Mat::zeros(rows * bloc, cols * bloc, CV_32S);

    for (int sy = 0; sy < rows; sy += superFactor)
    {
        for (int sx = 0; sx < cols; sx += superFactor)
        {
            std::map<int, int> counts;
            for (int dy = 0; dy < superFactor; ++dy)
                for (int dx = 0; dx < superFactor; ++dx)
                {
                    int y = sy + dy, x = sx + dx;
                    if (y < rows && x < cols)
                        counts[labels[y][x]]++;
                }

            int bestLabel = 0, bestCount = 0;
            for (auto &p : counts)
                if (p.second > bestCount)
                {
                    bestLabel = p.first;
                    bestCount = p.second;
                }

            // si homogène -> placer marqueur
            if (bestCount >= superFactor * superFactor * 0.8) // 80% homogène
            {
                int cx = (sx + superFactor / 2) * bloc;
                int cy = (sy + superFactor / 2) * bloc;
                if (bestLabel > 0)
                    markers.at<int>(cy, cx) = bestLabel;
                else if (bestLabel == 0 && counts.size() == 1)
                    markers.at<int>(cy, cx) = 0;
            }
        }
    }

    // petite dilatation pour agrandir les marqueurs
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(markers, markers, kernel);

    return markers;
}
