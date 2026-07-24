using System;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace CardCropperNet
{
    /// <summary>
    /// 🔥 基于 DocAligner (LCNet100) ONNX 模型的证件四角检测器。
    /// 这是裁剪质量的核心：AI 精准定位卡片四角，再在原图上做透视矫正。
    /// 完全离线，无需任何云端 OCR / 密钥 / 网络。
    /// </summary>
    public sealed class AiCornerDetector : IDisposable
    {
        private const int InferSize = 256;

        private readonly InferenceSession? _session;
        private readonly string _inputName = "img";
        private readonly string _outputName = "output_0";

        private static AiCornerDetector? _instance;
        public static AiCornerDetector Instance => _instance ??= new AiCornerDetector();

        public bool Available => _session != null;

        private AiCornerDetector()
        {
            try
            {
                string modelPath = Path.Combine(AppContext.BaseDirectory, "models", "docaligner_lcnet100.onnx");
                if (File.Exists(modelPath))
                {
                    var opts = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                    _session = new InferenceSession(modelPath, opts);
                    _inputName = _session.InputMetadata.Keys.First();
                    _outputName = _session.OutputMetadata.Keys.First();
                    TencentOCR.Log($"✅ DocAligner AI 模型已加载 (输入:{_inputName} 输出:{_outputName})");
                }
                else
                {
                    TencentOCR.Log($"⚠️ 未找到 AI 模型: {modelPath}，将使用传统边缘检测");
                }
            }
            catch (Exception ex)
            {
                _session = null;
                TencentOCR.Log($"❌ AI 模型加载失败: {ex.Message}");
            }
        }

        /// <summary>
        /// 检测卡片四角（原图坐标）。返回 (四角[左上,右上,右下,左下], 置信度)，失败返回 null。
        /// </summary>
        public (PointF[] corners, float confidence)? DetectCorners(Mat inputBgr)
        {
            if (_session == null) return null;

            try
            {
                int origRows = inputBgr.Rows;
                int origCols = inputBgr.Cols;

                // 1) 暗底反相判断（深色背景翻转成浅色，帮助模型定位）
                using var grayCheck = new Mat();
                CvInvoke.CvtColor(inputBgr, grayCheck, ColorConversion.Bgr2Gray);
                bool inverted = CvInvoke.Mean(grayCheck).V0 < 100.0;

                Mat working = inputBgr;
                Mat? invMat = null;
                if (inverted)
                {
                    invMat = new Mat();
                    CvInvoke.BitwiseNot(inputBgr, invMat);
                    working = invMat;
                }

                // 2) 灰度 + CLAHE 增强对比度，再转回 3 通道
                using var gray = new Mat();
                CvInvoke.CvtColor(working, gray, ColorConversion.Bgr2Gray);
                using var claheImg = new Mat();
                CvInvoke.CLAHE(gray, 2.0, new Size(8, 8), claheImg);
                using var enhanced = new Mat();
                CvInvoke.CvtColor(claheImg, enhanced, ColorConversion.Gray2Bgr);

                // 3) letterbox 缩放到 256×256（保持长宽比，居中填黑）
                int ec = enhanced.Cols, er = enhanced.Rows;
                float scale = Math.Min((float)InferSize / ec, (float)InferSize / er);
                int newW = (int)Math.Round(ec * scale);
                int newH = (int)Math.Round(er * scale);
                int padX = (InferSize - newW) / 2;
                int padY = (InferSize - newH) / 2;

                using var resized = new Mat();
                CvInvoke.Resize(enhanced, resized, new Size(newW, newH), 0, 0, Inter.Linear);
                using var canvas = new Mat(new Size(InferSize, InferSize), DepthType.Cv8U, 3);
                canvas.SetTo(new MCvScalar(0, 0, 0));
                var roi = new Mat(canvas, new System.Drawing.Rectangle(padX, padY, newW, newH));
                resized.CopyTo(roi);
                roi.Dispose();

                // 4) BGR→RGB，归一化到 [0,1]，填入 NCHW 张量
                using var rgb = new Mat();
                CvInvoke.CvtColor(canvas, rgb, ColorConversion.Bgr2Rgb);

                var tensor = new DenseTensor<float>(new[] { 1, 3, InferSize, InferSize });
                var rgbData = new byte[InferSize * InferSize * 3];
                Marshal.Copy(rgb.DataPointer, rgbData, 0, rgbData.Length);
                for (int y = 0; y < InferSize; y++)
                {
                    for (int x = 0; x < InferSize; x++)
                    {
                        int idx = (y * InferSize + x) * 3;
                        tensor[0, 0, y, x] = rgbData[idx] / 255f;     // R
                        tensor[0, 1, y, x] = rgbData[idx + 1] / 255f; // G
                        tensor[0, 2, y, x] = rgbData[idx + 2] / 255f; // B
                    }
                }

                invMat?.Dispose();

                // 5) 推理
                using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) });
                var output = results.First().AsTensor<float>();
                int[] dims = output.Dimensions.ToArray(); // [1, C, H, W]
                int channels = dims[1];
                int hmH = dims[2];
                int hmW = dims[3];
                if (channels < 4) return null;

                // 6) 每个通道取热力图峰值 + 局部加权质心，映射回原图坐标
                var corners = new PointF[4];
                for (int k = 0; k < 4; k++)
                {
                    float best = float.MinValue;
                    int bx = 0, by = 0;
                    for (int yy = 0; yy < hmH; yy++)
                    {
                        for (int xx = 0; xx < hmW; xx++)
                        {
                            float v = output[0, k, yy, xx];
                            if (v > best) { best = v; bx = xx; by = yy; }
                        }
                    }
                    if (best < 0.1f) return null;

                    // 峰值周围 ±3 加权质心，亚像素精度
                    const int win = 3;
                    float wsum = 0, wx = 0, wy = 0;
                    for (int dy = -win; dy <= win; dy++)
                    {
                        for (int dx = -win; dx <= win; dx++)
                        {
                            int ny = by + dy, nx = bx + dx;
                            if (ny < 0 || ny >= hmH || nx < 0 || nx >= hmW) continue;
                            float v = output[0, k, ny, nx];
                            if (v <= 0f) continue;
                            wsum += v; wx += v * nx; wy += v * ny;
                        }
                    }
                    float cx = wsum > 0 ? wx / wsum : bx;
                    float cy = wsum > 0 ? wy / wsum : by;

                    // 热力图坐标 → 256 画布坐标 → 去 letterbox → 原图坐标
                    float px256 = (cx + 0.5f) / hmW * InferSize;
                    float py256 = (cy + 0.5f) / hmH * InferSize;
                    float ox = (px256 - padX) / scale;
                    float oy = (py256 - padY) / scale;
                    ox = Math.Max(0, Math.Min(origCols - 1, ox));
                    oy = Math.Max(0, Math.Min(origRows - 1, oy));
                    corners[k] = new PointF(ox, oy);
                }

                var ordered = OrderCorners(corners);
                float conf = ComputeConfidence(ordered, origCols, origRows);
                if (conf < 0.3f) return null;

                return (ordered, conf);
            }
            catch (Exception ex)
            {
                TencentOCR.Log($"❌ AI 角点检测异常: {ex.Message}");
                return null;
            }
        }

        // 四角排序为 [左上, 右上, 右下, 左下]
        private static PointF[] OrderCorners(PointF[] pts)
        {
            var bySum = pts.OrderBy(p => p.X + p.Y).ToArray();
            PointF tl = bySum[0];
            PointF br = bySum[3];
            var mid = new[] { bySum[1], bySum[2] }.OrderBy(p => p.Y - p.X).ToArray();
            PointF tr = mid[0];
            PointF bl = mid[1];
            return new[] { tl, tr, br, bl };
        }

        private static float Distance(PointF a, PointF b)
        {
            float dx = a.X - b.X, dy = a.Y - b.Y;
            return (float)Math.Sqrt(dx * dx + dy * dy);
        }

        // 置信度：凸四边形 + 长宽比合理 + 在图内 + 对边长度接近
        private static float ComputeConfidence(PointF[] c, int imgW, int imgH)
        {
            if (!IsConvexQuad(c)) return 0f;

            float top = Distance(c[0], c[1]);
            float bottom = Distance(c[3], c[2]);
            float left = Distance(c[0], c[3]);
            float right = Distance(c[1], c[2]);
            float avgW = (top + bottom) / 2f;
            float avgH = (left + right) / 2f;
            float ratio = avgH > 0 ? avgW / avgH : 0;

            // 卡片横放 1.3~2.0，竖放 0.5~0.8
            if (!((ratio >= 1.3f && ratio <= 2.0f) || (ratio >= 0.5f && ratio <= 0.8f)))
                return 0.2f;

            float margin = 0.05f;
            float minX = -imgW * margin, maxX = imgW * (1 + margin);
            float minY = -imgH * margin, maxY = imgH * (1 + margin);
            foreach (var p in c)
                if (p.X < minX || p.X > maxX || p.Y < minY || p.Y > maxY) return 0.3f;

            float wDiff = Math.Abs(top - bottom) / Math.Max(top, bottom);
            float hDiff = Math.Abs(left - right) / Math.Max(left, right);
            float quality = 1f - (wDiff + hDiff) / 2f;
            return Math.Max(0.4f, quality);
        }

        private static bool IsConvexQuad(PointF[] pts)
        {
            if (pts.Length != 4) return false;
            float c0 = Cross(pts[0], pts[1], pts[2]);
            float c1 = Cross(pts[1], pts[2], pts[3]);
            float c2 = Cross(pts[2], pts[3], pts[0]);
            float c3 = Cross(pts[3], pts[0], pts[1]);
            if (c0 > 0 && c1 > 0 && c2 > 0 && c3 > 0) return true;
            if (c0 < 0 && c1 < 0 && c2 < 0 && c3 < 0) return true;
            return false;

            static float Cross(PointF a, PointF b, PointF c)
                => (b.X - a.X) * (c.Y - a.Y) - (b.Y - a.Y) * (c.X - a.X);
        }

        public void Dispose() => _session?.Dispose();
    }
}
