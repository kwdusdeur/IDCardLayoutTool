using System;
using System.Drawing;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace CardCropperNet
{
    public class CardCropper
    {
        private string cardType;
        private double expectedRatio;

        public CardCropper(string type)
        {
            cardType = type;
            expectedRatio = type switch
            {
                "身份证" => 85.6 / 54.0,
                "银行卡" => 85.6 / 53.98,
                "驾驶证" => 88.0 / 60.0,
                "护照" => 125.0 / 88.0,
                _ => 1.6
            };
        }

        public (Mat?, double) CropCard(Mat image)
        {
            try
            {
                // 🔥 7种裁剪策略，从强到弱依次尝试
                var methods = new (string name, Func<Mat, (Mat?, double)> method)[]
                {
                    ("多尺度Canny边缘", MethodMultiScaleCanny),
                    ("霍夫直线组合", MethodHoughLines),
                    ("增强边缘检测", MethodEnhancedEdges),
                    ("自适应二值化", MethodAdaptiveThreshold),
                    ("色彩分割", MethodColorSegmentation),
                    ("形态学闭操作", MethodMorphClose),
                    ("最小外接矩形", MethodMinAreaRect)
                };

                Mat? bestResult = null;
                double bestScore = 0;
                string bestMethod = "";

                foreach (var (name, method) in methods)
                {
                    try
                    {
                        var (result, score) = method(image);
                        
                        if (result != null && score > bestScore)
                        {
                            bestResult?.Dispose();
                            bestResult = result;
                            bestScore = score;
                            bestMethod = name;
                        }
                        else
                        {
                            result?.Dispose();
                        }

                        // 如果已经很好了，不再尝试后续方法
                        if (bestScore > 0.90) break;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"{name}失败: {ex.Message}");
                    }
                }

                if (bestResult != null && bestScore > 0.35)
                {
                    Console.WriteLine($"✅ 使用方法: {bestMethod}, 置信度: {bestScore:F2}");
                    return (bestResult, bestScore);
                }

                // 保底
                Console.WriteLine("⚠️ 所有方法失败，使用保底裁剪");
                return FallbackCrop(image);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"裁剪失败：{ex.Message}");
                return FallbackCrop(image);
            }
        }

        // 🔥 方法1: 多尺度Canny边缘检测（应对不同对比度）
        private (Mat?, double) MethodMultiScaleCanny(Mat image)
        {
            var gray = new Mat();
            CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);

            // 增强对比度
            var clahe = new Mat();
            CvInvoke.CLAHE(gray, 3.0, new Size(8, 8), clahe);

            var blurred = new Mat();
            CvInvoke.GaussianBlur(clahe, blurred, new Size(5, 5), 0);

            // 🔥 尝试5组不同阈值（扩大范围）
            var thresholds = new[] { (20, 80), (30, 100), (50, 150), (70, 200), (100, 250) };
            
            foreach (var (low, high) in thresholds)
            {
                var edges = new Mat();
                CvInvoke.Canny(blurred, edges, low, high);

                var kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1));
                var dilated = new Mat();
                CvInvoke.Dilate(edges, dilated, kernel, new Point(-1, -1), 2, BorderType.Default, new MCvScalar());

                var result = FindBestQuadrilateral(image, dilated);
                
                edges.Dispose();
                dilated.Dispose();
                kernel.Dispose();

                if (result.Item1 != null)
                {
                    gray.Dispose();
                    clahe.Dispose();
                    blurred.Dispose();
                    return result;
                }
            }

            gray.Dispose();
            clahe.Dispose();
            blurred.Dispose();
            return (null, 0);
        }

        // 🔥 方法2: 霍夫直线检测组合四边形
        private (Mat?, double) MethodHoughLines(Mat image)
        {
            var gray = new Mat();
            CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);

            var blurred = new Mat();
            CvInvoke.GaussianBlur(gray, blurred, new Size(5, 5), 0);

            var edges = new Mat();
            CvInvoke.Canny(blurred, edges, 50, 150);

            var lines = new VectorOfPointF();
            CvInvoke.HoughLinesP(edges, lines, 1, Math.PI / 180, 80, 50, 10);

            if (lines.Size >= 4)
            {
                // 按长度排序取前10条
                var lineArray = new LineSegment2D[lines.Size];
                for (int i = 0; i < lines.Size; i++)
                {
                    var p = lines[i];
                    lineArray[i] = new LineSegment2D(new Point((int)p.X, (int)p.Y), 
                        new Point((int)(p.X + 1), (int)(p.Y + 1)));
                }

                var sorted = lineArray.OrderByDescending(l => l.Length).Take(10).ToArray();

                // 尝试从直线中找交点组成四边形
                var intersections = new System.Collections.Generic.List<PointF>();
                for (int i = 0; i < sorted.Length - 1; i++)
                {
                    for (int j = i + 1; j < sorted.Length; j++)
                    {
                        var pt = LineIntersection(sorted[i], sorted[j]);
                        if (pt.HasValue && IsPointInImage(pt.Value, image))
                        {
                            intersections.Add(pt.Value);
                        }
                    }
                }

                if (intersections.Count >= 4)
                {
                    // 找外围4个点
                    var corners = FindOuterCorners(intersections.ToArray());
                    if (corners != null)
                    {
                        var warped = FourPointTransform(image, corners);
                        if (warped.Width < warped.Height)
                        {
                            var rotated = new Mat();
                            CvInvoke.Rotate(warped, rotated, RotateFlags.Rotate90Clockwise);
                            warped.Dispose();
                            warped = rotated;
                        }

                        var score = CheckAspectRatio(warped);
                        gray.Dispose();
                        blurred.Dispose();
                        edges.Dispose();
                        lines.Dispose();

                        if (score > 0.5)
                            return (warped, score);
                        
                        warped.Dispose();
                    }
                }
            }

            gray.Dispose();
            blurred.Dispose();
            edges.Dispose();
            lines.Dispose();
            return (null, 0);
        }

        // 🔥 方法3: 增强边缘检测（原方法加强版）
        private (Mat?, double) MethodEnhancedEdges(Mat image)
        {
            var gray = new Mat();
            CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);

            var clahe = new Mat();
            CvInvoke.CLAHE(gray, 3.0, new Size(8, 8), clahe);

            var blurred = new Mat();
            CvInvoke.GaussianBlur(clahe, blurred, new Size(5, 5), 0);

            var edges = new Mat();
            double median = GetMedian(blurred);
            int lower = (int)Math.Max(0, 0.66 * median);
            int upper = (int)Math.Min(255, 1.33 * median);
            CvInvoke.Canny(blurred, edges, lower, upper);

            var kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1));
            var dilated = new Mat();
            CvInvoke.Dilate(edges, dilated, kernel, new Point(-1, -1), 2, BorderType.Default, new MCvScalar());

            var result = FindBestQuadrilateral(image, dilated);

            gray.Dispose();
            clahe.Dispose();
            blurred.Dispose();
            edges.Dispose();
            kernel.Dispose();
            dilated.Dispose();

            return result;
        }

        // 🔥 方法4: 自适应二值化
        private (Mat?, double) MethodAdaptiveThreshold(Mat image)
        {
            var gray = new Mat();
            CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);

            var binary = new Mat();
            CvInvoke.AdaptiveThreshold(gray, binary, 255, AdaptiveThresholdType.GaussianC,
                ThresholdType.BinaryInv, 11, 2);

            var kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(9, 9), new Point(-1, -1));
            var closed = new Mat();
            CvInvoke.MorphologyEx(binary, closed, MorphOp.Close, kernel, new Point(-1, -1), 2, BorderType.Default, new MCvScalar());

            var result = FindBestQuadrilateral(image, closed);

            gray.Dispose();
            binary.Dispose();
            kernel.Dispose();
            closed.Dispose();

            return result;
        }

        // 🔥 方法5: 色彩分割（适合背景色与卡片色差大的场景）
        private (Mat?, double) MethodColorSegmentation(Mat image)
        {
            var hsv = new Mat();
            CvInvoke.CvtColor(image, hsv, ColorConversion.Bgr2Hsv);

            var channels = new VectorOfMat();
            CvInvoke.Split(hsv, channels);

            // 对饱和度和明度通道二值化
            var maskS = new Mat();
            var maskV = new Mat();
            CvInvoke.Threshold(channels[1], maskS, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);
            CvInvoke.Threshold(channels[2], maskV, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

            var combined = new Mat();
            CvInvoke.BitwiseOr(maskS, maskV, combined);

            var kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(7, 7), new Point(-1, -1));
            var closed = new Mat();
            CvInvoke.MorphologyEx(combined, closed, MorphOp.Close, kernel, new Point(-1, -1), 3, BorderType.Default, new MCvScalar());

            var result = FindBestQuadrilateral(image, closed);

            hsv.Dispose();
            channels.Dispose();
            maskS.Dispose();
            maskV.Dispose();
            combined.Dispose();
            kernel.Dispose();
            closed.Dispose();

            return result;
        }

        // 🔥 方法6: 形态学闭操作强化
        private (Mat?, double) MethodMorphClose(Mat image)
        {
            var gray = new Mat();
            CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);

            var binary = new Mat();
            CvInvoke.Threshold(gray, binary, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

            var kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(15, 15), new Point(-1, -1));
            var closed = new Mat();
            CvInvoke.MorphologyEx(binary, closed, MorphOp.Close, kernel, new Point(-1, -1), 3, BorderType.Default, new MCvScalar());

            var result = FindBestQuadrilateral(image, closed);

            gray.Dispose();
            binary.Dispose();
            kernel.Dispose();
            closed.Dispose();

            return result;
        }

        // 🔥 方法7: 最小外接矩形（最稳妥的保底方法）
        private (Mat?, double) MethodMinAreaRect(Mat image)
        {
            var gray = new Mat();
            CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);

            var binary = new Mat();
            CvInvoke.Threshold(gray, binary, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

            using (var contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(binary, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                if (contours.Size == 0)
                {
                    gray.Dispose();
                    binary.Dispose();
                    return (null, 0);
                }

                // 找最大轮廓
                int maxIdx = 0;
                double maxArea = 0;
                for (int i = 0; i < contours.Size; i++)
                {
                    var area = CvInvoke.ContourArea(contours[i]);
                    if (area > maxArea)
                    {
                        maxArea = area;
                        maxIdx = i;
                    }
                }

                var imgArea = image.Width * image.Height;
                if (maxArea < imgArea * 0.05)
                {
                    gray.Dispose();
                    binary.Dispose();
                    return (null, 0);
                }

                var rect = CvInvoke.MinAreaRect(contours[maxIdx]);
                var box = CvInvoke.BoxPoints(rect);
                var ordered = OrderPoints(box);
                var warped = FourPointTransform(image, ordered);

                if (warped.Width < warped.Height)
                {
                    var rotated = new Mat();
                    CvInvoke.Rotate(warped, rotated, RotateFlags.Rotate90Clockwise);
                    warped.Dispose();
                    warped = rotated;
                }

                var score = CheckAspectRatio(warped);

                gray.Dispose();
                binary.Dispose();

                return (warped, score);
            }
        }

        // 🔥 通用四边形查找逻辑（被多个方法复用）
        private (Mat?, double) FindBestQuadrilateral(Mat image, Mat edges)
        {
            using (var contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(edges, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                if (contours.Size == 0)
                    return (null, 0);

                var imgArea = image.Width * image.Height;

                // 按面积排序，取前5个
                var candidates = new System.Collections.Generic.List<(int idx, double area)>();
                for (int i = 0; i < contours.Size; i++)
                {
                    var area = CvInvoke.ContourArea(contours[i]);
                    if (area > imgArea * 0.05 && area < imgArea * 0.95)
                        candidates.Add((i, area));
                }

                candidates = candidates.OrderByDescending(c => c.area).Take(5).ToList();

                foreach (var (idx, area) in candidates)
                {
                    var contour = contours[idx];
                    var peri = CvInvoke.ArcLength(contour, true);

                    // 尝试多个epsilon值
                    foreach (var epsilon in new[] { 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06 })
                    {
                        using (var approx = new VectorOfPoint())
                        {
                            CvInvoke.ApproxPolyDP(contour, approx, epsilon * peri, true);

                            if (approx.Size == 4)
                            {
                                var points = approx.ToArray();
                                var ordered = OrderPoints(points);
                                var warped = FourPointTransform(image, ordered);

                                if (warped.Width < warped.Height)
                                {
                                    var rotated = new Mat();
                                    CvInvoke.Rotate(warped, rotated, RotateFlags.Rotate90Clockwise);
                                    warped.Dispose();
                                    warped = rotated;
                                }

                                var score = CheckAspectRatio(warped);

                                if (score > 0.5)
                                    return (warped, score);

                                warped.Dispose();
                            }
                        }
                    }
                }
            }

            return (null, 0);
        }

        private (Mat, double) FallbackCrop(Mat image)
        {
            // 保底：裁掉 5% 边距
            int marginX = (int)(image.Width * 0.05);
            int marginY = (int)(image.Height * 0.05);
            
            var cropped = new Mat(image, new Rectangle(marginX, marginY,
                image.Width - 2 * marginX, image.Height - 2 * marginY));
            
            return (cropped.Clone(), 0.3);
        }

        // 🔥 辅助函数：直线交点
        private PointF? LineIntersection(LineSegment2D line1, LineSegment2D line2)
        {
            double x1 = line1.P1.X, y1 = line1.P1.Y;
            double x2 = line1.P2.X, y2 = line1.P2.Y;
            double x3 = line2.P1.X, y3 = line2.P1.Y;
            double x4 = line2.P2.X, y4 = line2.P2.Y;

            double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
            if (Math.Abs(denom) < 1e-6) return null;

            double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
            double u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;

            if (t >= 0 && t <= 1 && u >= 0 && u <= 1)
            {
                double x = x1 + t * (x2 - x1);
                double y = y1 + t * (y2 - y1);
                return new PointF((float)x, (float)y);
            }

            return null;
        }

        private bool IsPointInImage(PointF pt, Mat image)
        {
            return pt.X >= 0 && pt.X < image.Width && pt.Y >= 0 && pt.Y < image.Height;
        }

        private PointF[]? FindOuterCorners(PointF[] points)
        {
            if (points.Length < 4) return null;

            // 找最外围4个点：左上/右上/右下/左下
            var tl = points.OrderBy(p => p.X + p.Y).First();
            var br = points.OrderByDescending(p => p.X + p.Y).First();
            var tr = points.OrderBy(p => p.Y - p.X).First();
            var bl = points.OrderByDescending(p => p.Y - p.X).First();

            return new[] { tl, tr, br, bl };
        }

        private Point[] OrderPoints(Point[] pts)
        {
            if (pts.Length < 4) return pts;

            var ordered = new Point[4];

            var sums = new double[pts.Length];
            for (int i = 0; i < pts.Length; i++)
                sums[i] = pts[i].X + pts[i].Y;
            int minIdx = Array.IndexOf(sums, sums.Min());
            ordered[0] = pts[minIdx];

            int maxIdx = Array.IndexOf(sums, sums.Max());
            ordered[2] = pts[maxIdx];

            var diffs = new double[pts.Length];
            for (int i = 0; i < pts.Length; i++)
                diffs[i] = pts[i].Y - pts[i].X;
            int minDiffIdx = Array.IndexOf(diffs, diffs.Min());
            ordered[1] = pts[minDiffIdx];

            int maxDiffIdx = Array.IndexOf(diffs, diffs.Max());
            ordered[3] = pts[maxDiffIdx];

            return ordered;
        }

        private PointF[] OrderPoints(PointF[] pts)
        {
            if (pts.Length < 4) return pts;

            var intPts = new Point[pts.Length];
            for (int i = 0; i < pts.Length; i++)
                intPts[i] = new Point((int)pts[i].X, (int)pts[i].Y);
            
            var ordered = OrderPoints(intPts);
            var result = new PointF[4];
            for (int i = 0; i < 4; i++)
                result[i] = new PointF(ordered[i].X, ordered[i].Y);
            
            return result;
        }

        private Mat FourPointTransform(Mat image, Point[] pts)
        {
            var ptsF = new PointF[pts.Length];
            for (int i = 0; i < pts.Length; i++)
                ptsF[i] = new PointF(pts[i].X, pts[i].Y);
            return FourPointTransform(image, ptsF);
        }

        private Mat FourPointTransform(Mat image, PointF[] pts)
        {
            if (pts.Length < 4) return image.Clone();

            var (tl, tr, br, bl) = (pts[0], pts[1], pts[2], pts[3]);

            // 🔥 轻微向外扩展四角，修复边缘多裁约1mm的问题
            var cx = (tl.X + tr.X + br.X + bl.X) / 4f;
            var cy = (tl.Y + tr.Y + br.Y + bl.Y) / 4f;
            const float expand = 1.005f; // 向外抓0.5%
            PointF Ex(PointF p) => new PointF(cx + (p.X - cx) * expand, cy + (p.Y - cy) * expand);
            tl = Ex(tl); tr = Ex(tr); br = Ex(br); bl = Ex(bl);

            double widthA = Distance(br, bl);
            double widthB = Distance(tr, tl);
            int maxWidth = (int)Math.Max(widthA, widthB);

            double heightA = Distance(tr, br);
            double heightB = Distance(tl, bl);
            int maxHeight = (int)Math.Max(heightA, heightB);

            if (maxWidth < 1 || maxHeight < 1)
                return image.Clone();

            var src = new PointF[] { tl, tr, br, bl };
            var dst = new PointF[] {
                new PointF(0, 0),
                new PointF(maxWidth - 1, 0),
                new PointF(maxWidth - 1, maxHeight - 1),
                new PointF(0, maxHeight - 1)
            };

            var matrix = CvInvoke.GetPerspectiveTransform(src, dst);
            var warped = new Mat();
            CvInvoke.WarpPerspective(image, warped, matrix, new Size(maxWidth, maxHeight));
            matrix.Dispose();

            // 🔥 后处理：自动对比度/亮度 + 圆角清理
            var cleaned = CleanupCard(warped);
            if (cleaned != warped) warped.Dispose();
            return cleaned;
        }

        // 🔥 后处理：自动对比度拉伸 + 四个圆角填白（更接近复印件）
        private Mat CleanupCard(Mat card)
        {
            if (card.Width < 10 || card.Height < 10) return card;

            // 1) 轻微自动白平衡：只把偏灰背景提亮到接近白，绝不压暗
            var outImg = card.Clone();
            try
            {
                var ch = new VectorOfMat();
                CvInvoke.Split(outImg, ch);
                for (int c = 0; c < ch.Size; c++)
                {
                    double minVal = 0, maxVal = 0;
                    System.Drawing.Point minLoc = default, maxLoc = default;
                    CvInvoke.MinMaxLoc(ch[c], ref minVal, ref maxVal, ref minLoc, ref maxLoc);
                    // 仅当白点明显低于255（背景偏灰）时提亮，且限制倍数避免过曝
                    if (maxVal > 180 && maxVal < 250)
                    {
                        double scale = 255.0 / maxVal;
                        if (scale > 1.25) scale = 1.25;
                        ch[c].ConvertTo(ch[c], DepthType.Cv8U, scale, 0);
                    }
                }
                CvInvoke.Merge(ch, outImg);
                ch.Dispose();
            }
            catch { }

            // 2) 四个圆角填白（半径约为短边的4.5%，接近真实证卡圆角）
            try
            {
                int r = (int)(Math.Min(outImg.Width, outImg.Height) * 0.045);
                if (r > 2)
                {
                    var white = new MCvScalar(255, 255, 255);
                    int w = outImg.Width, h = outImg.Height;
                    // 左上
                    FillCorner(outImg, 0, 0, r, white, true, true);
                    FillCorner(outImg, w, 0, r, white, false, true);
                    FillCorner(outImg, 0, h, r, white, true, false);
                    FillCorner(outImg, w, h, r, white, false, false);
                }
            }
            catch { }

            return outImg;
        }

        // 填充单个圆角（在角落矩形区域内，圆弧外部填白）
        private void FillCorner(Mat img, int cornerX, int cornerY, int r, MCvScalar color, bool left, bool top)
        {
            int ccx = left ? r : cornerX - r;
            int ccy = top ? r : cornerY - r;
            int startX = left ? 0 : cornerX - r;
            int startY = top ? 0 : cornerY - r;
            for (int y = 0; y < r; y++)
            {
                int py = startY + y;
                if (py < 0 || py >= img.Height) continue;
                double dy = py - ccy;
                // 求该行圆弧内的水平跨度
                double inside = (double)r * r - dy * dy;
                if (inside < 0) inside = 0;
                int halfSpan = (int)Math.Sqrt(inside);
                // 圆弧外部区域（需填白）
                if (left)
                {
                    int fillTo = ccx - halfSpan; // 从 startX 到 fillTo-1 填白
                    int w = fillTo - startX;
                    if (w > 0) CvInvoke.Rectangle(img, new Rectangle(startX, py, w, 1), color, -1);
                }
                else
                {
                    int fillFrom = ccx + halfSpan;
                    int w = (startX + r) - fillFrom;
                    if (w > 0) CvInvoke.Rectangle(img, new Rectangle(fillFrom, py, w, 1), color, -1);
                }
            }
        }

        private double Distance(Point p1, Point p2)
        {
            return Math.Sqrt(Math.Pow(p2.X - p1.X, 2) + Math.Pow(p2.Y - p1.Y, 2));
        }

        private double Distance(PointF p1, PointF p2)
        {
            return Math.Sqrt(Math.Pow(p2.X - p1.X, 2) + Math.Pow(p2.Y - p1.Y, 2));
        }

        private double CheckAspectRatio(Mat image)
        {
            if (image.Height == 0) return 0;
            double ratio = (double)image.Width / image.Height;
            double diff = Math.Abs(ratio - expectedRatio);
            return Math.Max(0, 1 - diff / expectedRatio);
        }

        private double GetMedian(Mat mat)
        {
            try
            {
                int total = mat.Rows * mat.Cols;
                var values = new byte[total];
                System.Runtime.InteropServices.Marshal.Copy(mat.DataPointer, values, 0, total);
                Array.Sort(values);
                return values[values.Length / 2];
            }
            catch
            {
                return 128;
            }
        }

        // 🔥 图像调整：亮度/对比度/阴影/高光
        public Mat AdjustTones(Mat image, int brightness, int contrast, int shadow, int highlight)
        {
            if (brightness == 0 && contrast == 0 && shadow == 0 && highlight == 0)
                return image.Clone();

            // 先处理亮度/对比度（在BGR上）
            var work = new Mat();
            double alpha = 1.0 + contrast / 100.0;   // 对比度系数 0~2
            double beta = brightness * 1.27;          // 亮度偏移 -127~127
            CvInvoke.ConvertScaleAbs(image, work, alpha, beta);

            // 再处理阴影/高光（在L通道上用LUT）
            if (shadow != 0 || highlight != 0)
            {
                var lut = new byte[256];
                double sh = shadow / 100.0;
                double hi = highlight / 100.0;
                for (int i = 0; i < 256; i++)
                {
                    double v = i / 255.0;
                    double shadowWeight = Math.Pow(1 - v, 2);
                    v += sh * 0.5 * shadowWeight;
                    double highWeight = Math.Pow(v, 2);
                    v += hi * 0.5 * highWeight;
                    int outv = (int)Math.Round(v * 255);
                    lut[i] = (byte)Math.Max(0, Math.Min(255, outv));
                }

                var lab = new Mat();
                CvInvoke.CvtColor(work, lab, ColorConversion.Bgr2Lab);
                var ch = new VectorOfMat();
                CvInvoke.Split(lab, ch);

                using (var lutMat = new Mat(1, 256, DepthType.Cv8U, 1))
                {
                    System.Runtime.InteropServices.Marshal.Copy(lut, 0, lutMat.DataPointer, 256);
                    CvInvoke.LUT(ch[0], lutMat, ch[0]);
                }

                var merged = new Mat();
                CvInvoke.Merge(ch, merged);
                var outImg = new Mat();
                CvInvoke.CvtColor(merged, outImg, ColorConversion.Lab2Bgr);

                lab.Dispose();
                ch.Dispose();
                merged.Dispose();
                work.Dispose();
                return outImg;
            }

            return work;
        }

        // 🔥 转黑白
        public static Mat ToGrayscale(Mat image)
        {
            var gray = new Mat();
            CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);
            var bgr = new Mat();
            CvInvoke.CvtColor(gray, bgr, ColorConversion.Gray2Bgr);
            gray.Dispose();
            return bgr;
        }
    }
}
