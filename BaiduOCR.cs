using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Emgu.CV;
using System.Windows;

namespace CardCropperNet
{
    public class BaiduOCR
    {
        private static readonly string API_KEY = "xHfCv6RsMUVhtb5Y7aTsBVLK";
        private static readonly string SECRET_KEY = "LoGb5NPZLObL4IU6uewEIvtzSxhY3SC5";
        private static string? accessToken = null;
        private static DateTime tokenExpiry = DateTime.MinValue;

        // 获取Access Token（带缓存）
        private static async Task<string> GetAccessToken()
        {
            if (accessToken != null && DateTime.Now < tokenExpiry)
                return accessToken;

            using var client = new HttpClient();
            var url = $"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_KEY}&client_secret={SECRET_KEY}";
            
            var response = await client.GetStringAsync(url);
            var json = JsonDocument.Parse(response);
            
            accessToken = json.RootElement.GetProperty("access_token").GetString();
            var expiresIn = json.RootElement.GetProperty("expires_in").GetInt32();
            tokenExpiry = DateTime.Now.AddSeconds(expiresIn - 600); // 提前10分钟刷新
            
            Console.WriteLine($"✅ 百度OCR Token已获取，有效期{expiresIn}秒");
            return accessToken!;
        }

        // 身份证识别（带裁剪）
        public static async Task<(Mat? croppedImage, double confidence)> RecognizeIdCard(Mat image, string side = "front")
        {
            try
            {
                Console.WriteLine($"🌐 调用百度OCR识别身份证...");
                var token = await GetAccessToken();
                var base64 = MatToBase64(image);

                using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
                
                // 🔥 使用通用文字识别（高精度版）获取文字位置
                var url = $"https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token={token}";

                var content = new StringContent(
                    $"image={Uri.EscapeDataString(base64)}&detect_direction=true&probability=true",
                    Encoding.UTF8,
                    "application/x-www-form-urlencoded"
                );

                var response = await client.PostAsync(url, content);
                var result = await response.Content.ReadAsStringAsync();
                
                Console.WriteLine($"百度API响应: {result.Substring(0, Math.Min(200, result.Length))}...");
                
                var json = JsonDocument.Parse(result);

                // 检查是否有错误
                if (json.RootElement.TryGetProperty("error_code", out var errCode))
                {
                    var errMsg = json.RootElement.TryGetProperty("error_msg", out var msg) ? msg.GetString() : "未知错误";
                    Console.WriteLine($"❌ 百度API错误 [{errCode}]: {errMsg}");
                    return (null, 0);
                }

                // 提取所有文字区域的边界框
                if (json.RootElement.TryGetProperty("words_result", out var words) && words.GetArrayLength() > 0)
                {
                    int minX = int.MaxValue, minY = int.MaxValue;
                    int maxX = 0, maxY = 0;
                    double totalProb = 0;
                    int count = 0;

                    foreach (var word in words.EnumerateArray())
                    {
                        if (word.TryGetProperty("location", out var loc))
                        {
                            int left = loc.GetProperty("left").GetInt32();
                            int top = loc.GetProperty("top").GetInt32();
                            int width = loc.GetProperty("width").GetInt32();
                            int height = loc.GetProperty("height").GetInt32();

                            minX = Math.Min(minX, left);
                            minY = Math.Min(minY, top);
                            maxX = Math.Max(maxX, left + width);
                            maxY = Math.Max(maxY, top + height);
                        }

                        if (word.TryGetProperty("probability", out var prob))
                        {
                            totalProb += prob.GetProperty("average").GetDouble();
                            count++;
                        }
                    }

                    if (count > 0)
                    {
                        // 扩展边距15%（留白）
                        int marginX = (int)((maxX - minX) * 0.15);
                        int marginY = (int)((maxY - minY) * 0.15);

                        minX = Math.Max(0, minX - marginX);
                        minY = Math.Max(0, minY - marginY);
                        maxX = Math.Min(image.Width, maxX + marginX);
                        maxY = Math.Min(image.Height, maxY + marginY);

                        var rect = new System.Drawing.Rectangle(minX, minY, maxX - minX, maxY - minY);
                        
                        // 检查裁剪区域是否合理
                        if (rect.Width < 50 || rect.Height < 50 || rect.Width > image.Width * 0.95 || rect.Height > image.Height * 0.95)
                        {
                            Console.WriteLine($"⚠️ 百度OCR裁剪区域不合理: {rect.Width}x{rect.Height}");
                            return (null, 0);
                        }

                        var cropped = new Mat(image, rect).Clone();

                        // 横向检查：身份证应该是横向的
                        if (cropped.Height > cropped.Width)
                        {
                            var rotated = new Mat();
                            CvInvoke.Rotate(cropped, rotated, Emgu.CV.CvEnum.RotateFlags.Rotate90Clockwise);
                            cropped.Dispose();
                            cropped = rotated;
                        }

                        double confidence = totalProb / count;
                        Console.WriteLine($"✅ 百度OCR识别成功: {cropped.Width}x{cropped.Height}, 置信度={confidence:F2}");
                        return (cropped, confidence);
                    }
                }

                Console.WriteLine($"⚠️ 百度OCR未检测到文字区域");
                return (null, 0);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ 百度OCR异常: {ex.Message}");
                return (null, 0);
            }
        }

        // Mat转Base64
        private static string MatToBase64(Mat mat)
        {
            using var vec = new Emgu.CV.Util.VectorOfByte();
            CvInvoke.Imencode(".jpg", mat, vec);
            var bytes = vec.ToArray();
            return Convert.ToBase64String(bytes);
        }
    }
}
