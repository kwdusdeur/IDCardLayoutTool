using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using Microsoft.Win32;
using Emgu.CV;
using Emgu.CV.Structure;
using PdfSharp.Pdf;
using PdfSharp.Drawing;
using System.Drawing.Printing;

namespace CardCropperNet
{
    public partial class MainWindow : Window
    {
        private ObservableCollection<ImageItem> imageItems = new ObservableCollection<ImageItem>();
        private CardCropper? cropper;
        private ImageItem? currentItem;
        private List<Mat> layoutPages = new List<Mat>();
        private int currentPageIndex = 0;

        private bool isDragging = false;
        private int dragStartIndex = -1;
        private Line? dragSeparatorLine;

        private List<ImageItem> selectionOrder = new List<ImageItem>();

        public MainWindow()
        {
            InitializeComponent();
            ImageListBox.ItemsSource = imageItems;
            cropper = new CardCropper("身份证");

            RadioIdCard.Checked += (s, e) => cropper = new CardCropper("身份证");
            RadioBankCard.Checked += (s, e) => cropper = new CardCropper("银行卡");
            RadioDriverLicense.Checked += (s, e) => cropper = new CardCropper("驾驶证");
            RadioPassport.Checked += (s, e) => cropper = new CardCropper("护照");
        }

        private List<ImageItem> GetTargetItems()
        {
            var sel = selectionOrder.Where(i => imageItems.Contains(i)).ToList();
            return sel.Count > 0 ? sel : imageItems.ToList();
        }

        private void AddImages_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFileDialog
            {
                Multiselect = true,
                Filter = "图片文件|*.jpg;*.jpeg;*.png;*.bmp|所有文件|*.*"
            };

            if (dialog.ShowDialog() == true)
            {
                foreach (var file in dialog.FileNames)
                    AddImage(file);
            }
        }

        private void AddImage(string filePath)
        {
            try
            {
                var item = new ImageItem
                {
                    FilePath = filePath,
                    FileName = System.IO.Path.GetFileName(filePath),
                    Index = imageItems.Count
                };
                item.GenerateThumbnail();
                imageItems.Add(item);
                UpdateIndices();
                StatusText.Text = $"已添加 {imageItems.Count} 张图片";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"添加图片失败：{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void UpdateIndices()
        {
            for (int i = 0; i < imageItems.Count; i++)
                imageItems[i].Index = i;
        }

        private void ImageListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            foreach (ImageItem removed in e.RemovedItems.OfType<ImageItem>())
                selectionOrder.Remove(removed);
            foreach (ImageItem added in e.AddedItems.OfType<ImageItem>())
                if (!selectionOrder.Contains(added))
                    selectionOrder.Add(added);

            if (ImageListBox.SelectedItems.Count == 1 && ImageListBox.SelectedItem is ImageItem item)
            {
                currentItem = item;
                ShowPreview(item);
            }
            else if (ImageListBox.SelectedItems.Count > 1)
            {
                StatusText.Text = $"已选中 {ImageListBox.SelectedItems.Count} 张（操作只针对选中）";
            }
            else if (ImageListBox.SelectedItems.Count == 0)
            {
                StatusText.Text = "";
            }
        }

        // 点击空白处取消选中
        private void ImageListBox_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            var elem = e.OriginalSource as DependencyObject;
            while (elem != null && elem != ImageListBox)
            {
                if (elem is ListBoxItem) return;
                elem = VisualTreeHelper.GetParent(elem);
            }
            ImageListBox.UnselectAll();
        }

        // 右键框选多张照片
        private void ImageListBox_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            var elem = e.OriginalSource as DependencyObject;
            ListBoxItem? targetItem = null;
            
            while (elem != null && elem != ImageListBox)
            {
                if (elem is ListBoxItem lbi)
                {
                    targetItem = lbi;
                    break;
                }
                elem = VisualTreeHelper.GetParent(elem);
            }

            if (targetItem != null && targetItem.Content is ImageItem clickedItem)
            {
                // 如果已选中，扩展选区；否则开始新选区
                if (!ImageListBox.SelectedItems.Contains(clickedItem))
                {
                    ImageListBox.SelectedItems.Clear();
                    ImageListBox.SelectedItems.Add(clickedItem);
                }
                else
                {
                    // 右键已选中的项时，保持多选状态
                    e.Handled = true;
                }
            }
        }

        private void ShowPreview(ImageItem item)
        {
            try
            {
                ClearLayoutPages();
                // 裁剪预览时隐藏上下页按钮
                PageControlPanel.Visibility = Visibility.Collapsed;

                if (item.CroppedImage != null)
                {
                    PreviewImage.Source = ImageItem.MatToBitmapSource(item.CroppedImage);
                    StatusText.Text = $"预览：{item.FileName}（已裁剪，置信度 {item.Confidence:P0}）";
                }
                else
                {
                    var mat = CvInvoke.Imread(item.FilePath, Emgu.CV.CvEnum.ImreadModes.Color);
                    PreviewImage.Source = ImageItem.MatToBitmapSource(mat);
                    StatusText.Text = $"预览：{item.FileName}（原图）";
                    mat.Dispose();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"预览失败：{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        // ============ 多页预览 ============
        private void ShowPage(int index)
        {
            if (layoutPages.Count == 0 || index < 0 || index >= layoutPages.Count) return;
            currentPageIndex = index;
            PreviewImage.Source = ImageItem.MatToBitmapSource(layoutPages[index]);
            PageInfoText.Text = $"第 {index + 1} 页 / 共 {layoutPages.Count} 页";
            PrevPageBtn.IsEnabled = index > 0;
            NextPageBtn.IsEnabled = index < layoutPages.Count - 1;
            // 拼版预览时显示上下页按钮
            PageControlPanel.Visibility = Visibility.Visible;
        }

        private void PrevPage_Click(object sender, RoutedEventArgs e)
        {
            if (currentPageIndex > 0) ShowPage(currentPageIndex - 1);
        }

        private void NextPage_Click(object sender, RoutedEventArgs e)
        {
            if (currentPageIndex < layoutPages.Count - 1) ShowPage(currentPageIndex + 1);
        }

        // ============ 一键裁剪（带透视纠正） ============
        private void AutoCrop_Click(object sender, RoutedEventArgs e)
        {
            if (imageItems.Count == 0)
            {
                MessageBox.Show("请先添加图片！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var targets = GetTargetItems();

            try
            {
                int successCount = 0, lowConfidenceCount = 0;

                foreach (var item in targets)
                {
                    var mat = item.OriginalImage ?? CvInvoke.Imread(item.FilePath, Emgu.CV.CvEnum.ImreadModes.Color);

                    if (cropper != null)
                    {
                        var (croppedMat, confidence) = cropper.CropCard(mat);
                        if (croppedMat != null)
                        {
                            item.CroppedImage?.Dispose();
                            item.CroppedImage = croppedMat.Clone();
                            item.Confidence = confidence;
                            item.RefreshThumbnail();
                            successCount++;
                            if (confidence < 0.8) lowConfidenceCount++;
                        }
                        croppedMat?.Dispose();
                    }

                    if (item.OriginalImage == null) mat.Dispose();
                }

                bool usedSelection = selectionOrder.Count(i => imageItems.Contains(i)) > 0;
                CropStatusText.Text = $"✅ 裁剪完成：{successCount}/{targets.Count}" + (usedSelection ? "（选中）" : "（全部）");
                if (lowConfidenceCount > 0)
                    CropStatusText.Text += $"\n⚠️ {lowConfidenceCount} 张置信度较低，建议手动裁剪";

                if (currentItem != null) ShowPreview(currentItem);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"裁剪失败：{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        // 手动裁剪
        private void ManualCrop_Click(object sender, RoutedEventArgs e)
        {
            if (currentItem == null)
            {
                MessageBox.Show("请先选择一张图片！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            try
            {
                var mat = currentItem.OriginalImage ?? CvInvoke.Imread(currentItem.FilePath, Emgu.CV.CvEnum.ImreadModes.Color);
                var win = new ManualCropWindow(mat) { Owner = this };

                if (win.ShowDialog() == true && win.ResultImage != null)
                {
                    currentItem.CroppedImage?.Dispose();
                    currentItem.CroppedImage = win.ResultImage.Clone();
                    currentItem.Confidence = 1.0;
                    currentItem.RefreshThumbnail();
                    ShowPreview(currentItem);
                    CropStatusText.Text = "✅ 手动裁剪完成";
                }

                if (currentItem.OriginalImage == null) mat.Dispose();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"手动裁剪失败：{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        // ============ 明暗度调整 ============
        private void AdjustmentSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (currentItem?.CroppedImage == null || cropper == null) return;

            try
            {
                var adjusted = cropper.AdjustTones(currentItem.CroppedImage,
                    (int)HighlightSlider.Value, (int)MidtoneSlider.Value, (int)ShadowSlider.Value);
                PreviewImage.Source = ImageItem.MatToBitmapSource(adjusted);
                adjusted.Dispose();
            }
            catch { }
        }

        private void ApplyAdjustments_Click(object sender, RoutedEventArgs e)
        {
            if (cropper == null) return;

            int h = (int)HighlightSlider.Value, m = (int)MidtoneSlider.Value, s = (int)ShadowSlider.Value;
            if (h == 0 && m == 0 && s == 0)
            {
                MessageBox.Show("三个滑块都是 0，没有需要应用的调整。", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var targets = GetTargetItems().Where(i => i.CroppedImage != null).ToList();
            if (targets.Count == 0)
            {
                MessageBox.Show("目标图片还没裁剪，请先裁剪再调整。", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            try
            {
                foreach (var item in targets)
                {
                    var adjusted = cropper.AdjustTones(item.CroppedImage!, h, m, s);
                    item.CroppedImage!.Dispose();
                    item.CroppedImage = adjusted;
                    item.RefreshThumbnail();
                }

                HighlightSlider.Value = 0;
                MidtoneSlider.Value = 0;
                ShadowSlider.Value = 0;

                bool usedSelection = selectionOrder.Count(i => imageItems.Contains(i)) > 0;
                CropStatusText.Text = $"✅ 已应用明暗度到 {targets.Count} 张" + (usedSelection ? "（选中）" : "（全部）");
                if (currentItem != null) ShowPreview(currentItem);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"应用调整失败：{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void ResetAdjustments_Click(object sender, RoutedEventArgs e)
        {
            HighlightSlider.Value = 0;
            MidtoneSlider.Value = 0;
            ShadowSlider.Value = 0;
            if (currentItem != null) ShowPreview(currentItem);
        }

        // ============ A4 拼版（多页） ============
        private void AutoLayout_Click(object sender, RoutedEventArgs e)
        {
            if (imageItems.Count == 0)
            {
                MessageBox.Show("请先添加并裁剪图片！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var source = GetTargetItems();
            var mats = new List<Mat>();
            foreach (var item in source)
            {
                if (item.CroppedImage != null)
                    mats.Add(item.CroppedImage);
            }

            if (mats.Count == 0)
            {
                MessageBox.Show("目标图片都还没裁剪，请先「一键裁剪」！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            try
            {
                var cardType = RadioIdCard.IsChecked == true ? "身份证" :
                               RadioBankCard.IsChecked == true ? "银行卡" :
                               RadioDriverLicense.IsChecked == true ? "驾驶证" : "护照";

                ClearLayoutPages();

                for (int i = 0; i < mats.Count; i += 2)
                {
                    var front = mats[i];
                    var back = (i + 1 < mats.Count) ? mats[i + 1] : null;
                    layoutPages.Add(A4Layout.Compose(front, back, cardType));
                }

                ShowPage(0);
                bool usedSelection = selectionOrder.Count(x => imageItems.Contains(x)) > 0;
                StatusText.Text = $"A4 拼版：{layoutPages.Count} 页（300 DPI，间隔 55mm）" +
                                  (usedSelection ? "，按选中顺序配对" : "，按 1-2/3-4 顺序");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"拼版失败：{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        // ============ 打印/导出 ============
        private void Print_Click(object sender, RoutedEventArgs e)
        {
            if (layoutPages.Count == 0)
            {
                MessageBox.Show("请先拼版！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            try
            {
                int pageIdx = 0;
                var pd = new PrintDocument();
                pd.PrintPage += (s, ev) =>
                {
                    using var bmp = layoutPages[pageIdx].ToBitmap();
                    ev.Graphics?.DrawImage(bmp, ev.MarginBounds);
                    pageIdx++;
                    ev.HasMorePages = pageIdx < layoutPages.Count;
                };

                var dlg = new PrintDialog();
                if (dlg.ShowDialog() == true)
                    pd.Print();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"打印失败：{ex.Message}\n\n提示：可先导出 PDF，再用 PDF 阅读器打印。", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void ExportPdf_Click(object sender, RoutedEventArgs e)
        {
            if (layoutPages.Count == 0)
            {
                MessageBox.Show("请先拼版！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var dialog = new SaveFileDialog { Filter = "PDF 文件|*.pdf", FileName = "证卡拼版.pdf" };
            if (dialog.ShowDialog() == true)
            {
                try
                {
                    using var doc = new PdfDocument();
                    foreach (var pageMat in layoutPages)
                    {
                        var page = doc.AddPage();
                        page.Width = XUnit.FromMillimeter(210);
                        page.Height = XUnit.FromMillimeter(297);

                        using var gfx = XGraphics.FromPdfPage(page);
                        using var bmp = pageMat.ToBitmap();
                        using var ms = new MemoryStream();
                        bmp.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                        ms.Position = 0;
                        using var ximg = XImage.FromStream(ms);
                        gfx.DrawImage(ximg, 0, 0, page.Width.Point, page.Height.Point);
                    }

                    doc.Save(dialog.FileName);
                    MessageBox.Show($"导出 PDF 成功（{layoutPages.Count} 页）！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"导出 PDF 失败：{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void ExportImage_Click(object sender, RoutedEventArgs e)
        {
            if (layoutPages.Count > 0)
            {
                var dialog = new SaveFileDialog
                {
                    Filter = "JPEG 图片|*.jpg|PNG 图片|*.png",
                    FileName = "A4拼版.jpg"
                };
                if (dialog.ShowDialog() == true)
                {
                    try
                    {
                        if (layoutPages.Count == 1)
                        {
                            CvInvoke.Imwrite(dialog.FileName, layoutPages[0]);
                        }
                        else
                        {
                            var dir = System.IO.Path.GetDirectoryName(dialog.FileName)!;
                            var baseName = System.IO.Path.GetFileNameWithoutExtension(dialog.FileName);
                            var ext = System.IO.Path.GetExtension(dialog.FileName);
                            for (int i = 0; i < layoutPages.Count; i++)
                                CvInvoke.Imwrite(System.IO.Path.Combine(dir, $"{baseName}_{i + 1}{ext}"), layoutPages[i]);
                        }
                        MessageBox.Show($"导出成功（{layoutPages.Count} 张）！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"导出失败：{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                }
                return;
            }

            var toExport = currentItem?.CroppedImage;
            if (toExport == null)
            {
                MessageBox.Show("请先裁剪或拼版！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var d2 = new SaveFileDialog
            {
                Filter = "JPEG 图片|*.jpg|PNG 图片|*.png",
                FileName = $"裁剪_{currentItem?.FileName}"
            };
            if (d2.ShowDialog() == true)
            {
                try
                {
                    CvInvoke.Imwrite(d2.FileName, toExport);
                    MessageBox.Show("导出成功！", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"导出失败：{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        // ============ 列表操作 ============
        private void RemoveImage_Click(object sender, RoutedEventArgs e)
        {
            var selected = ImageListBox.SelectedItems.Cast<ImageItem>().ToList();
            if (selected.Count == 0) return;

            foreach (var item in selected)
            {
                selectionOrder.Remove(item);
                item.Dispose();
                imageItems.Remove(item);
            }
            UpdateIndices();
            StatusText.Text = $"已删除 {selected.Count} 张，剩余 {imageItems.Count} 张";
        }

        private void ClearList_Click(object sender, RoutedEventArgs e)
        {
            foreach (var item in imageItems)
                item.Dispose();
            imageItems.Clear();
            selectionOrder.Clear();
            PreviewImage.Source = null;
            StatusText.Text = "";
            CropStatusText.Text = "";
            PageInfoText.Text = "";
            ClearLayoutPages();
        }

        private void ClearLayoutPages()
        {
            foreach (var p in layoutPages) p.Dispose();
            layoutPages.Clear();
            currentPageIndex = 0;
            PageInfoText.Text = "";
            PrevPageBtn.IsEnabled = false;
            NextPageBtn.IsEnabled = false;
            PageControlPanel.Visibility = Visibility.Collapsed;
        }

        // 旋转
        private void RotateCW_Click(object sender, RoutedEventArgs e) => RotateSelected(true);
        private void RotateCCW_Click(object sender, RoutedEventArgs e) => RotateSelected(false);

        private void RotateSelected(bool clockwise)
        {
            var selected = ImageListBox.SelectedItems.Cast<ImageItem>().ToList();
            if (selected.Count == 0)
            {
                MessageBox.Show("请先选中要旋转的图片。", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            foreach (var item in selected)
                item.Rotate(clockwise);

            if (currentItem != null && selected.Contains(currentItem))
                ShowPreview(currentItem);
            StatusText.Text = $"已{(clockwise ? "顺" : "逆")}时针旋转 {selected.Count} 张";
        }

        // ============ 拖拽排序（带分割线提示） ============
        private void ImageListBox_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (e.OriginalSource is FrameworkElement elem && elem.DataContext is ImageItem item)
                dragStartIndex = imageItems.IndexOf(item);
        }

        private void ImageListBox_PreviewMouseMove(object sender, MouseEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed && dragStartIndex >= 0 && !isDragging)
            {
                isDragging = true;
                var item = imageItems[dragStartIndex];
                DragDrop.DoDragDrop(ImageListBox, item, DragDropEffects.Move);
                RemoveDragSeparator();
                isDragging = false;
                dragStartIndex = -1;
            }
        }

        private void ImageListBox_Drop(object sender, DragEventArgs e)
        {
            RemoveDragSeparator();
            if (e.Data.GetData(typeof(ImageItem)) is ImageItem draggedItem)
            {
                var pos = e.GetPosition(ImageListBox);
                var target = GetItemAtPosition(pos);
                if (target != null && target != draggedItem)
                {
                    int oldIdx = imageItems.IndexOf(draggedItem);
                    int newIdx = imageItems.IndexOf(target);
                    imageItems.Move(oldIdx, newIdx);
                    UpdateIndices();
                }
            }
        }

        private void ImageListBox_DragOver(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(typeof(ImageItem)))
            {
                e.Effects = DragDropEffects.Move;
                ShowDragSeparator(e.GetPosition(ImageListBox));
            }
            e.Handled = true;
        }

        private void ShowDragSeparator(Point pos)
        {
            RemoveDragSeparator();
            var target = GetItemAtPosition(pos);
            if (target == null) return;

            var container = ImageListBox.ItemContainerGenerator.ContainerFromItem(target) as FrameworkElement;
            if (container == null) return;

            var itemPos = container.TransformToAncestor(ImageListBox).Transform(new Point(0, 0));
            dragSeparatorLine = new Line
            {
                X1 = 0,
                X2 = ImageListBox.ActualWidth,
                Y1 = itemPos.Y,
                Y2 = itemPos.Y,
                Stroke = Brushes.Blue,
                StrokeThickness = 3
            };

            var adornerLayer = AdornerLayer.GetAdornerLayer(ImageListBox);
            if (adornerLayer != null)
            {
                var adorner = new LineAdorner(ImageListBox, dragSeparatorLine);
                adornerLayer.Add(adorner);
            }
        }

        private void RemoveDragSeparator()
        {
            if (dragSeparatorLine == null) return;
            var adornerLayer = AdornerLayer.GetAdornerLayer(ImageListBox);
            if (adornerLayer != null)
            {
                var adorners = adornerLayer.GetAdorners(ImageListBox);
                if (adorners != null)
                {
                    foreach (var adorner in adorners.OfType<LineAdorner>())
                        adornerLayer.Remove(adorner);
                }
            }
            dragSeparatorLine = null;
        }

        private ImageItem? GetItemAtPosition(Point pos)
        {
            var elem = ImageListBox.InputHitTest(pos) as DependencyObject;
            while (elem != null && elem != ImageListBox)
            {
                if (elem is ListBoxItem lbi) return lbi.Content as ImageItem;
                elem = VisualTreeHelper.GetParent(elem);
            }
            return null;
        }

        // ============ 键盘快捷键 ============
        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Delete && ImageListBox.SelectedItems.Count > 0)
                RemoveImage_Click(sender, e);
            else if (e.Key == Key.V && (Keyboard.Modifiers & ModifierKeys.Control) == ModifierKeys.Control)
                PasteFromClipboard();
        }

        private void PasteFromClipboard()
        {
            try
            {
                if (Clipboard.ContainsImage())
                {
                    var bmpSrc = Clipboard.GetImage();
                    if (bmpSrc == null) return;

                    var encoder = new PngBitmapEncoder();
                    encoder.Frames.Add(BitmapFrame.Create(bmpSrc));
                    var tempPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"paste_{Guid.NewGuid()}.png");
                    using (var fs = new FileStream(tempPath, FileMode.Create))
                        encoder.Save(fs);

                    AddImage(tempPath);
                }
                else if (Clipboard.ContainsFileDropList())
                {
                    foreach (var file in Clipboard.GetFileDropList())
                        if (file != null && IsImageFile(file))
                            AddImage(file);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"粘贴失败：{ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void Window_DragOver(object sender, DragEventArgs e)
        {
            e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop) ? DragDropEffects.Copy : DragDropEffects.None;
            e.Handled = true;
        }

        private void Window_Drop(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                string[] files = (string[])e.Data.GetData(DataFormats.FileDrop);
                foreach (var file in files)
                    if (IsImageFile(file)) AddImage(file);
            }
        }

        private bool IsImageFile(string file)
        {
            var ext = System.IO.Path.GetExtension(file).ToLower();
            return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
        }
    }

    // 拖拽分割线 Adorner
    public class LineAdorner : Adorner
    {
        private Line line;

        public LineAdorner(UIElement adornedElement, Line line) : base(adornedElement)
        {
            this.line = line;
        }

        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawLine(new Pen(line.Stroke, line.StrokeThickness),
                new Point(line.X1, line.Y1), new Point(line.X2, line.Y2));
        }
    }
}
