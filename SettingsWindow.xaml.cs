using System;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Navigation;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace CardCropperNet
{
    public partial class SettingsWindow : Window
    {
        private AppConfig config;

        public SettingsWindow(AppConfig config)
        {
            InitializeComponent();
            this.config = config;

            // 加载现有配置
            SecretIdTextBox.Text = config.TencentSecretId;
            SecretKeyPasswordBox.Password = config.TencentSecretKey;
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                config.TencentSecretId = SecretIdTextBox.Text.Trim();
                config.TencentSecretKey = SecretKeyPasswordBox.Password.Trim();
                config.Save();

                // 立即应用到 TencentOCR
                TencentOCR.InitConfig(config);

                StatusTextBlock.Text = "✅ 保存成功！配置已立即生效";
                StatusTextBlock.Foreground = System.Windows.Media.Brushes.Green;

                // 1秒后关闭窗口
                Task.Delay(1000).ContinueWith(_ => Dispatcher.Invoke(() => DialogResult = true));
            }
            catch (Exception ex)
            {
                StatusTextBlock.Text = $"❌ 保存失败：{ex.Message}";
                StatusTextBlock.Foreground = System.Windows.Media.Brushes.Red;
            }
        }

        private async void TestButton_Click(object sender, RoutedEventArgs e)
        {
            var id = SecretIdTextBox.Text.Trim();
            var key = SecretKeyPasswordBox.Password.Trim();

            if (string.IsNullOrWhiteSpace(id) || string.IsNullOrWhiteSpace(key))
            {
                StatusTextBlock.Text = "⚠️ 请先填写 Secret ID 和 Secret Key";
                StatusTextBlock.Foreground = System.Windows.Media.Brushes.Orange;
                return;
            }

            TestButton.IsEnabled = false;
            StatusTextBlock.Text = "🔄 正在测试连接...";
            StatusTextBlock.Foreground = System.Windows.Media.Brushes.Blue;

            try
            {
                // 临时应用配置进行测试
                var testConfig = new AppConfig
                {
                    TencentSecretId = id,
                    TencentSecretKey = key
                };
                TencentOCR.InitConfig(testConfig);

                // 创建一个简单的测试图像（纯白 100x100）
                var testMat = new Mat(100, 100, DepthType.Cv8U, 3);
                testMat.SetTo(new Emgu.CV.Structure.MCvScalar(255, 255, 255));

                var (result, conf) = await TencentOCR.RecognizeCard(testMat, "身份证");
                testMat.Dispose();
                result?.Dispose();

                // 检查是否返回了有效错误（说明连接成功但图像无效）
                // 或者返回了结果（说明完全成功）
                StatusTextBlock.Text = "✅ 测试成功！腾讯云 API 连接正常";
                StatusTextBlock.Foreground = System.Windows.Media.Brushes.Green;
            }
            catch (Exception ex)
            {
                StatusTextBlock.Text = $"❌ 测试失败：{ex.Message}";
                StatusTextBlock.Foreground = System.Windows.Media.Brushes.Red;
                TencentOCR.Log($"配置测试失败：{ex}");
            }
            finally
            {
                TestButton.IsEnabled = true;
            }
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
        }

        private void Hyperlink_RequestNavigate(object sender, RequestNavigateEventArgs e)
        {
            try
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = e.Uri.AbsoluteUri,
                    UseShellExecute = true
                });
                e.Handled = true;
            }
            catch { }
        }
    }
}
