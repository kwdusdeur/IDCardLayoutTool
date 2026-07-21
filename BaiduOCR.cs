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
            tokenExpiry = DateTime.Now.AddSeconds(expiresIn - 600);
            
            Console.WriteLine($"✅ 百度OCR Token已获取，有效期{expiresIn}秒");
            return accessToken!;
        }

        // 🔥 身份证混贴识别（返回精确卡片位置）
        public static async Task<(Mat? croppedImage, double confidence)> RecognizeIdCard(Mat image)
        {
            try
            {
                Console.WriteLine($"🌐 调用百度身份证混贴识别API...");
                var token = await GetAccessToken();
                var base64 = MatToBase64(image);

                using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
                var url = $"https://aip.baidubce.com/rest/2.0/ocr/v1/multi_idcard?access_token={token}";

                var content = new StringContent(
                    $"image={Uri.EscapeDataString(base64)}&detect_direction=true",
                    Encoding.UTF8,
                    "application/x-www-form-urlencoded"
                );

                var response = await client.PostAsync(url, content);
                var result = await response.Content.ReadAsStringAsync();
                
                Console.WriteLine($"百度API响应前200字符: {result.Substring(0, Math.Min(200, result.Length))}...");
                
                var json = JsonDocument.Parse(result);

                // 检查错误
                if (json.RootElement.TryGetProperty("error_code", out var errCode))
                {
                    var errMsg = json.RootElement.TryGetProperty("error_msg", out var msg) ? msg.GetString() : "未知错误";
                    Console.WriteLine($"❌ 百度API错误 [{errCode}]: {errMsg}");
                    return (null, 0);
                }

                // 提取第一张身份证的位置
                if (json.RootElement.TryGetProperty("words_result", out var results) && results.GetArrayLength() > 0)
                {
                    var first = results[0];
                    if (first.TryGetProperty("card_info", out var cardInfo))
                    {
                        var cardType = cardInfo.TryGetProperty("card_type", out var ct) ? ct.GetString() : "unknown";
                        var imageStatus = cardInfo.TryGetProperty("image_status", out var ist) ? ist.GetString() : "unknown";
                        
                        // 非身份证类型
                        if (cardType == null || (!cardType.StartsWith("idcard_")))
                        {
                            Console.WriteLine($"⚠️ 百度检测到非身份证类型: {cardType}");
                            return (null, 0);
                        }

                        // 图片状态异常
                        if (imageStatus != "normal" && imageStatus != "reversed_side")
                        {
                            Console.WriteLine($"⚠️ 百度检测图片状态异常: {imageStatus}");
                            return (null, 0);
                        }

                        // 提取卡片位置
                        if (cardInfo.TryGetProperty("card_location", out var loc))
                        {
                            int left = loc.GetProperty("left").GetInt32();
                            int top = loc.GetProperty("top").GetInt32();
                            int width = loc.GetProperty("width").GetInt32();
                            int height = loc.GetProperty("height").GetInt32();

                            // 检查坐标合理性
                            if (left < 0 || top < 0 || width < 50 || height < 50 || 
                                left + width > image.Width || top + height > image.Height)
                            {
                                Console.WriteLine($"⚠️ 百度返回坐标异常: L={left},T={top},W={width},H={height}");
                                return (null, 0);
                            }

                            // 裁剪
                            var rect = new System.Drawing.Rectangle(left, top, width, height);
                            var cropped = new Mat(image, rect).Clone();

                            // 自动旋转：身份证应该是横向的
                            if (cropped.Height > cropped.Width)
                            {
                                var rotated = new Mat();
                                CvInvoke.Rotate(cropped, rotated, Emgu.CV.CvEnum.RotateFlags.Rotate90Clockwise);
                                cropped.Dispose();
                                cropped = rotated;
                            }

                            // 置信度：百度检测到身份证=高置信度
                            double confidence = 0.95;
                            Console.WriteLine($"✅ 百度OCR识别成功: 类型={cardType}, 状态={imageStatus}, 尺寸={cropped.Width}x{cropped.Height}");
                            return (cropped, confidence);
                        }
                        else
                        {
                            Console.WriteLine($"⚠️ 百度API未返回card_location");
                        }
                    }
                }

                Console.WriteLine($"⚠️ 百度API未检测到身份证");
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
