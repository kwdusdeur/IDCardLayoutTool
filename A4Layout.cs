using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace CardCropperNet
{
    public static class A4Layout
    {
        // A4 @ 300 DPI
        const int DPI = 300;
        const double MM_PER_INCH = 25.4;

        static int MmToPx(double mm) => (int)Math.Round(mm / MM_PER_INCH * DPI);

        // A4 尺寸：210 x 297 mm
        static readonly int A4_WIDTH = MmToPx(210);   // 2480
        static readonly int A4_HEIGHT = MmToPx(297);  // 3508

        // 卡片标准尺寸 85.6 x 54 mm
        static readonly int CARD_WIDTH = MmToPx(85.6);
        static readonly int CARD_HEIGHT = MmToPx(54.0);

        // 正反面间隔 55mm
        static readonly int GAP = MmToPx(55.0);

        /// <summary>
        /// 生成 A4 拼版：正面在上，反面在下，间隔 55mm，居中。
        /// front/back 为已裁剪的卡片图。back 可为 null。
        /// </summary>
        public static Mat Compose(Mat front, Mat? back, string cardType = "身份证")
        {
            var (cardW, cardH) = GetCardSize(cardType);

            // 白色 A4 底
            var canvas = new Mat(A4_HEIGHT, A4_WIDTH, DepthType.Cv8U, 3);
            canvas.SetTo(new MCvScalar(255, 255, 255));

            // 缩放正面到标准卡片尺寸
            var frontResized = ResizeToCard(front, cardW, cardH);

            // 计算垂直排布：两张卡 + 间隔，整体垂直居中
            int totalHeight = back != null ? (cardH * 2 + GAP) : cardH;
            int startY = (A4_HEIGHT - totalHeight) / 2;
            int startX = (A4_WIDTH - cardW) / 2;

            // 贴正面
            CopyInto(canvas, frontResized, startX, startY);
            frontResized.Dispose();

            // 贴反面
            if (back != null)
            {
                var backResized = ResizeToCard(back, cardW, cardH);
                int backY = startY + cardH + GAP;
                CopyInto(canvas, backResized, startX, backY);
                backResized.Dispose();
            }

            return canvas;
        }

        static (int, int) GetCardSize(string cardType)
        {
            return cardType switch
            {
                "身份证" => (MmToPx(85.6), MmToPx(54.0)),
                "银行卡" => (MmToPx(85.6), MmToPx(53.98)),
                "驾驶证" => (MmToPx(88.0), MmToPx(60.0)),
                "护照"   => (MmToPx(125.0), MmToPx(88.0)),
                _ => (MmToPx(85.6), MmToPx(54.0))
            };
        }

        static Mat ResizeToCard(Mat card, int w, int h)
        {
            var resized = new Mat();
            CvInvoke.Resize(card, resized, new Size(w, h), 0, 0, Inter.Cubic);
            return resized;
        }

        static void CopyInto(Mat canvas, Mat card, int x, int y)
        {
            // 边界保护
            if (x < 0) x = 0;
            if (y < 0) y = 0;
            int w = card.Width;
            int h = card.Height;
            if (x + w > canvas.Width) w = canvas.Width - x;
            if (y + h > canvas.Height) h = canvas.Height - y;
            if (w <= 0 || h <= 0) return;

            using var roi = new Mat(canvas, new Rectangle(x, y, w, h));
            if (w != card.Width || h != card.Height)
            {
                using var src = new Mat(card, new Rectangle(0, 0, w, h));
                src.CopyTo(roi);
            }
            else
            {
                card.CopyTo(roi);
            }
        }
    }
}
