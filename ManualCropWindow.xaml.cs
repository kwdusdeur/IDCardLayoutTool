using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace CardCropperNet
{
    public partial class ManualCropWindow : Window
    {
        private Mat sourceImage;
        private double displayScale = 1.0;
        private int imgDisplayW, imgDisplayH;
        private double currentZoom = 1.0;

        // 🔥 平移相关
        private bool isPanning = false;
        private Point lastPanPoint;

        private List<Point> clickedPoints = new List<Point>();
        private List<Ellipse> markers = new List<Ellipse>();
        private const double MARKER_R = 8;

        private static readonly string[] CornerNames =
        {
            "① 左上角", "② 右上角", "③ 右下角", "④ 左下角"
        };

        public Mat? ResultImage { get; private set; }
        public bool Confirmed { get; private set; }

        public ManualCropWindow(Mat image)
        {
            InitializeComponent();
            sourceImage = image;
            Loaded += (s, e) => SetupImage();
        }

        private void SetupImage()
        {
            double maxW = 860;
            double maxH = 560;

            displayScale = Math.Min(maxW / sourceImage.Width, maxH / sourceImage.Height);
            if (displayScale > 1) displayScale = 1;

            imgDisplayW = (int)(sourceImage.Width * displayScale);
            imgDisplayH = (int)(sourceImage.Height * displayScale);

            CropImage.Source = ImageItem.MatToBitmapSource(sourceImage);
            CropImage.Width = imgDisplayW;
            CropImage.Height = imgDisplayH;

            CropCanvas.Width = imgDisplayW;
            CropCanvas.Height = imgDisplayH;

            UpdateHint();
        }

        private void UpdateHint()
        {
            if (clickedPoints.Count < 4)
            {
                HintText.Text = $"请依次点击：{CornerNames[clickedPoints.Count]}";
                ConfirmBtn.IsEnabled = false;
            }
            else
            {
                HintText.Text = "✅ 四个角已选好，点「确认裁剪」，或「重新点选」";
                ConfirmBtn.IsEnabled = true;
            }
        }

        private void Canvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (clickedPoints.Count >= 4) return;

            var pos = e.GetPosition(CropCanvas);
            pos.X = Math.Max(0, Math.Min(imgDisplayW, pos.X));
            pos.Y = Math.Max(0, Math.Min(imgDisplayH, pos.Y));
            clickedPoints.Add(pos);

            var marker = new Ellipse
            {
                Width = MARKER_R * 2,
                Height = MARKER_R * 2,
                Fill = new SolidColorBrush(Color.FromRgb(0x3F, 0x8E, 0xFF)),
                Stroke = Brushes.White,
                StrokeThickness = 2
            };
            Canvas.SetLeft(marker, pos.X - MARKER_R);
            Canvas.SetTop(marker, pos.Y - MARKER_R);
            CropCanvas.Children.Add(marker);
            markers.Add(marker);

            var label = new TextBlock
            {
                Text = (clickedPoints.Count).ToString(),
                Foreground = Brushes.White,
                FontWeight = FontWeights.Bold,
                FontSize = 12
            };
            Canvas.SetLeft(label, pos.X - 4);
            Canvas.SetTop(label, pos.Y - 9);
            CropCanvas.Children.Add(label);

            UpdatePolygon();
            UpdateHint();
        }

        private void UpdatePolygon()
        {
            var pts = new System.Windows.Media.PointCollection();
            foreach (var c in clickedPoints) pts.Add(c);
            CropPolygon.Points = pts;
        }

        // 旋转原图
        private void RotateCW_Click(object sender, RoutedEventArgs e) => RotateImage(true);
        private void RotateCCW_Click(object sender, RoutedEventArgs e) => RotateImage(false);

        private void RotateImage(bool clockwise)
        {
            var rotated = new Mat();
            CvInvoke.Rotate(sourceImage, rotated,
                clockwise ? RotateFlags.Rotate90Clockwise : RotateFlags.Rotate90CounterClockwise);
            sourceImage.Dispose();
            sourceImage = rotated;

            // 重置所有点选
            Reset_Click(null, null);
            SetupImage();
        }

        // 🔥 鼠标滚轮缩放（PreviewMouseWheel 才能拦截 ScrollViewer 的滚动）
        private void Scroller_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            double delta = e.Delta > 0 ? 1.15 : 1.0 / 1.15;
            double newZoom = currentZoom * delta;
            newZoom = Math.Max(0.3, Math.Min(8.0, newZoom));
            ApplyZoom(newZoom);
            e.Handled = true;
        }

        private void Confirm_Click(object sender, RoutedEventArgs e)
        {
            if (clickedPoints.Count < 4) return;

            try
            {
                var src = new System.Drawing.PointF[4];
                for (int i = 0; i < 4; i++)
                {
                    src[i] = new System.Drawing.PointF(
                        (float)(clickedPoints[i].X / displayScale),
                        (float)(clickedPoints[i].Y / displayScale));
                }

                double widthTop = Dist(src[0], src[1]);
                double widthBot = Dist(src[3], src[2]);
                double heightL = Dist(src[0], src[3]);
                double heightR = Dist(src[1], src[2]);
                int maxW = (int)Math.Max(widthTop, widthBot);
                int maxH = (int)Math.Max(heightL, heightR);
                if (maxW < 1 || maxH < 1) return;

                var dst = new System.Drawing.PointF[]
                {
                    new System.Drawing.PointF(0, 0),
                    new System.Drawing.PointF(maxW - 1, 0),
                    new System.Drawing.PointF(maxW - 1, maxH - 1),
                    new System.Drawing.PointF(0, maxH - 1)
                };

                using var matrix = CvInvoke.GetPerspectiveTransform(src, dst);
                var warped = new Mat();
                CvInvoke.WarpPerspective(sourceImage, warped, matrix,
                    new System.Drawing.Size(maxW, maxH));

                // 自动旋转：横向卡片
                if (warped.Width < warped.Height)
                {
                    var rot = new Mat();
                    CvInvoke.Rotate(warped, rot, RotateFlags.Rotate90Clockwise);
                    warped.Dispose();
                    warped = rot;
                }

                ResultImage = warped;
                Confirmed = true;
                DialogResult = true;
                Close();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"裁剪失败：{ex.Message}", "错误");
            }
        }

        private double Dist(System.Drawing.PointF a, System.Drawing.PointF b)
            => Math.Sqrt(Math.Pow(a.X - b.X, 2) + Math.Pow(a.Y - b.Y, 2));

        private void Reset_Click(object sender, RoutedEventArgs e)
        {
            clickedPoints.Clear();
            foreach (var m in markers) CropCanvas.Children.Remove(m);
            markers.Clear();
            for (int i = CropCanvas.Children.Count - 1; i >= 0; i--)
            {
                if (CropCanvas.Children[i] is TextBlock)
                    CropCanvas.Children.RemoveAt(i);
            }
            CropPolygon.Points = new System.Windows.Media.PointCollection();
            UpdateHint();
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            Confirmed = false;
            DialogResult = false;
            Close();
        }

        // 🔥 放大按钮
        private void ZoomIn_Click(object sender, RoutedEventArgs e)
        {
            double newZoom = currentZoom * 1.2;
            newZoom = Math.Min(5.0, newZoom);
            ApplyZoom(newZoom);
        }

        // 🔥 缩小按钮
        private void ZoomOut_Click(object sender, RoutedEventArgs e)
        {
            double newZoom = currentZoom / 1.2;
            newZoom = Math.Max(0.3, newZoom);
            ApplyZoom(newZoom);
        }

        // 🔥 适应窗口
        private void ZoomFit_Click(object sender, RoutedEventArgs e)
        {
            ApplyZoom(1.0);
            ImageScroller.ScrollToHorizontalOffset(0);
            ImageScroller.ScrollToVerticalOffset(0);
        }

        private void ApplyZoom(double newZoom)
        {
            currentZoom = newZoom;
            CanvasScale.ScaleX = currentZoom;
            CanvasScale.ScaleY = currentZoom;
            ZoomText.Text = $"{(int)(currentZoom * 100)}%";
        }

        // 🔥 抓手工具或Ctrl+左键拖动平移
        private void Scroller_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            bool panMode = (PanToggle.IsChecked == true) ||
                           Keyboard.IsKeyDown(Key.LeftCtrl) || Keyboard.IsKeyDown(Key.RightCtrl) ||
                           e.MiddleButton == MouseButtonState.Pressed;
            if (panMode)
            {
                isPanning = true;
                lastPanPoint = e.GetPosition(ImageScroller);
                ImageScroller.Cursor = Cursors.Hand;
                e.Handled = true;
            }
        }

        private void Scroller_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (isPanning)
            {
                var currentPoint = e.GetPosition(ImageScroller);
                double deltaX = lastPanPoint.X - currentPoint.X;
                double deltaY = lastPanPoint.Y - currentPoint.Y;

                ImageScroller.ScrollToHorizontalOffset(ImageScroller.HorizontalOffset + deltaX);
                ImageScroller.ScrollToVerticalOffset(ImageScroller.VerticalOffset + deltaY);

                lastPanPoint = currentPoint;
                e.Handled = true;
            }
        }

        private void Scroller_PreviewMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (isPanning)
            {
                isPanning = false;
                ImageScroller.Cursor = Cursors.Arrow;
                e.Handled = true;
            }
        }
    }
}
