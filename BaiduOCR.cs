using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Emgu.CV;

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
            
            return accessToken!;
        }

        // 身份证识别（带裁剪）
        public static async Task<(Mat? croppedImage, double confidence)> RecognizeIdCard(Mat image, string side = "front")
        {
            try
            {
                var token = await GetAccessToken();
                var base64 = MatToBase64(image);

                using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
                var url = $"https://aip.baidubce.com/rest/2.0/ocr/v1/idcard?access_token={token}";

                var content = new StringContent(
                    $"id_card_side={side}&image={Uri.EscapeDataString(base64)}&detect_direction=true&detect_risk=false",
                    Encoding.UTF8,
                    "application/x-www-form-urlencoded"
                );

                var response = await client.PostAsync(url, content);
                var result = await response.Content.ReadAsStringAsync();
                var json = JsonDocument.Parse(result);

                // 检查是否有错误
                if (json.RootElement.TryGetProperty("error_code", out _))
                {
                    Console.WriteLine($"百度API错误: {result}");
                    return (null, 0);
                }

                // 提取裁剪区域（四角坐标）
                if (json.RootElement.TryGetProperty("image_status", out var status) && 
                    status.GetString() == "normal" &&
                    json.RootElement.TryGetProperty("words_result", out var words))
                {
                    // 百度返回的是识别结果，我们需要根据文字位置推算卡片范围
                    // 更好的方式是用身份证检测API
                    return await RecognizeIdCardWithDetect(image);
                }

                return (null, 0);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"百度OCR异常: {ex.Message}");
                return (null, 0);
            }
        }

        // 身份证检测（返回裁剪后图片）
        private static async Task<(Mat? croppedImage, double confidence)> RecognizeIdCardWithDetect(Mat image)
        {
            try
            {
                var token = await GetAccessToken();
                var base64 = MatToBase64(image);

                using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
                var url = $"https://aip.baidubce.com/rest/2.0/image-classify/v1/body_attr?access_token={token}";

                // 使用通用文字识别（带位置信息）
                url = $"https://aip.baidubce.com/rest/2.0/ocr/v1/general?access_token={token}";

                var content = new StringContent(
                    $"image={Uri.EscapeDataString(base64)}&detect_direction=true&paragraph=false&probability=true",
                    Encoding.UTF8,
                    "application/x-www-form-urlencoded"
                );

                var response = await client.PostAsync(url, content);
                var result = await response.Content.ReadAsStringAsync();
                var json = JsonDocument.Parse(result);

                if (json.RootElement.TryGetProperty("words_result", out var words) && words.GetArrayLength() > 0)
                {
                    // 提取所有文字区域的边界框
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
                        // 扩展边距10%
                        int marginX = (int)((maxX - minX) * 0.1);
                        int marginY = (int)((maxY - minY) * 0.1);

                        minX = Math.Max(0, minX - marginX);
                        minY = Math.Max(0, minY - marginY);
                        maxX = Math.Min(image.Width, maxX + marginX);
                        maxY = Math.Min(image.Height, maxY + marginY);

                        var rect = new System.Drawing.Rectangle(minX, minY, maxX - minX, maxY - minY);
                        var cropped = new Mat(image, rect).Clone();

                        // 横向检查：如果是竖的，旋转90度
                        if (cropped.Height > cropped.Width)
                        {
                            var rotated = new Mat();
                            CvInvoke.Rotate(cropped, rotated, Emgu.CV.CvEnum.RotateFlags.Rotate90Clockwise);
                            cropped.Dispose();
                            cropped = rotated;
                        }

                        double confidence = totalProb / count;
                        return (cropped, confidence);
                    }
                }

                return (null, 0);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"百度检测异常: {ex.Message}");
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
