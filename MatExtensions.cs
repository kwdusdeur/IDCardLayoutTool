using System.Drawing;
using System.Drawing.Imaging;
using Emgu.CV;

namespace CardCropperNet
{
    public static class MatExtensions
    {
        public static Bitmap ToBitmap(this Mat mat)
        {
            int width = mat.Width;
            int height = mat.Height;
            int channels = mat.NumberOfChannels;

            PixelFormat format;
            if (channels == 1)
                format = PixelFormat.Format8bppIndexed;
            else if (channels == 3)
                format = PixelFormat.Format24bppRgb;
            else if (channels == 4)
                format = PixelFormat.Format32bppArgb;
            else
                throw new System.NotSupportedException($"不支持 {channels} 通道图像");

            var bitmap = new Bitmap(width, height, format);

            // 灰度需要调色板
            if (channels == 1)
            {
                var pal = bitmap.Palette;
                for (int i = 0; i < 256; i++)
                {
                    pal.Entries[i] = Color.FromArgb(i, i, i);
                }
                bitmap.Palette = pal;
            }

            var bmpData = bitmap.LockBits(
                new Rectangle(0, 0, width, height),
                ImageLockMode.WriteOnly,
                format);

            int stride = (int)mat.Step;
            int bmpStride = bmpData.Stride;

            unsafe
            {
                byte* src = (byte*)mat.DataPointer.ToPointer();
                byte* dst = (byte*)bmpData.Scan0.ToPointer();

                for (int y = 0; y < height; y++)
                {
                    byte* srcRow = src + y * stride;
                    byte* dstRow = dst + y * bmpStride;

                    for (int x = 0; x < width * channels; x++)
                    {
                        dstRow[x] = srcRow[x];
                    }
                }
            }

            bitmap.UnlockBits(bmpData);
            return bitmap;
        }
    }
}
