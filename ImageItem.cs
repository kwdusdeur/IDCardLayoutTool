using System;
using System.ComponentModel;
using System.IO;
using System.Windows.Media.Imaging;
using Emgu.CV;
using Emgu.CV.CvEnum;

namespace CardCropperNet
{
    public class ImageItem : IDisposable, INotifyPropertyChanged
    {
        public string FilePath { get; set; } = "";

        private string _fileName = "";
        public string FileName
        {
            get => _fileName;
            set { _fileName = value; OnPropertyChanged(nameof(FileName)); }
        }

        public Mat? OriginalImage { get; set; }
        public Mat? CroppedImage { get; set; }
        public double Confidence { get; set; }

        private int _index;
        public int Index
        {
            get => _index;
            set { _index = value; OnPropertyChanged(nameof(DisplayLabel)); }
        }

        // 显示标签：照片1、照片2、照片3...
        public string DisplayLabel
        {
            get => $"照片 {Index + 1}";
        }

        private BitmapSource? _thumbnailImage;
        public BitmapSource? ThumbnailImage
        {
            get => _thumbnailImage;
            set { _thumbnailImage = value; OnPropertyChanged(nameof(ThumbnailImage)); }
        }

        public void GenerateThumbnail()
        {
            try
            {
                // 读取原图（若尚未读取）
                if (OriginalImage == null)
                {
                    OriginalImage = CvInvoke.Imread(FilePath, ImreadModes.Color);
                }
                RefreshThumbnail();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"生成缩略图失败：{ex.Message}");
            }
        }

        // 从当前 OriginalImage（或 CroppedImage）重新生成内存缩略图
        public void RefreshThumbnail()
        {
            try
            {
                var source = CroppedImage ?? OriginalImage;
                if (source == null) return;

                var thumbnail = new Mat();
                int maxWidth = 200;
                int maxHeight = 140;

                double scale = Math.Min((double)maxWidth / source.Width,
                                        (double)maxHeight / source.Height);
                int newWidth = Math.Max(1, (int)(source.Width * scale));
                int newHeight = Math.Max(1, (int)(source.Height * scale));

                CvInvoke.Resize(source, thumbnail, new System.Drawing.Size(newWidth, newHeight));
                ThumbnailImage = MatToBitmapSource(thumbnail);
                thumbnail.Dispose();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"刷新缩略图失败：{ex.Message}");
            }
        }

        // 旋转原图（正数=顺时针90，负数=逆时针90）
        public void Rotate(bool clockwise)
        {
            if (OriginalImage == null) return;
            var rotated = new Mat();
            CvInvoke.Rotate(OriginalImage, rotated,
                clockwise ? RotateFlags.Rotate90Clockwise : RotateFlags.Rotate90CounterClockwise);
            OriginalImage.Dispose();
            OriginalImage = rotated;

            // 若已裁剪，同步旋转裁剪结果
            if (CroppedImage != null)
            {
                var cr = new Mat();
                CvInvoke.Rotate(CroppedImage, cr,
                    clockwise ? RotateFlags.Rotate90Clockwise : RotateFlags.Rotate90CounterClockwise);
                CroppedImage.Dispose();
                CroppedImage = cr;
            }

            RefreshThumbnail();
        }

        public static BitmapSource MatToBitmapSource(Mat mat)
        {
            Mat bgr = mat;
            bool needDispose = false;
            if (mat.NumberOfChannels == 1)
            {
                bgr = new Mat();
                CvInvoke.CvtColor(mat, bgr, ColorConversion.Gray2Bgr);
                needDispose = true;
            }
            else if (mat.NumberOfChannels == 4)
            {
                bgr = new Mat();
                CvInvoke.CvtColor(mat, bgr, ColorConversion.Bgra2Bgr);
                needDispose = true;
            }

            int width = bgr.Width;
            int height = bgr.Height;
            int stride = (int)bgr.Step;
            int bufferSize = stride * height;

            var buffer = new byte[bufferSize];
            System.Runtime.InteropServices.Marshal.Copy(bgr.DataPointer, buffer, 0, bufferSize);

            var bmp = BitmapSource.Create(
                width, height, 96, 96,
                System.Windows.Media.PixelFormats.Bgr24,
                null, buffer, stride);
            bmp.Freeze();

            if (needDispose) bgr.Dispose();
            return bmp;
        }

        public event PropertyChangedEventHandler? PropertyChanged;
        protected void OnPropertyChanged(string name)
            => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));

        public void Dispose()
        {
            OriginalImage?.Dispose();
            CroppedImage?.Dispose();
        }
    }
}
