using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace CardCropperNet
{
    /// <summary>
    /// 应用配置（自动保存到 config.json）
    /// </summary>
    public class AppConfig
    {
        [JsonPropertyName("tencent_secret_id")]
        public string TencentSecretId { get; set; } = "";

        [JsonPropertyName("tencent_secret_key")]
        public string TencentSecretKey { get; set; } = "";

        [JsonPropertyName("output_dpi")]
        public int OutputDpi { get; set; } = 300;

        [JsonPropertyName("default_card_type")]
        public string DefaultCardType { get; set; } = "身份证";

        [JsonPropertyName("auto_save_path")]
        public string AutoSavePath { get; set; } = "";

        private static readonly string ConfigPath = Path.Combine(AppContext.BaseDirectory, "config.json");

        /// <summary>
        /// 加载配置文件，不存在时创建模板
        /// </summary>
        public static AppConfig Load()
        {
            try
            {
                if (!File.Exists(ConfigPath))
                {
                    TencentOCR.Log("⚠️ config.json 不存在，正在创建模板...");
                    var template = new AppConfig();
                    template.Save();
                    TencentOCR.Log($"✅ 已生成配置文件模板：{ConfigPath}");
                    TencentOCR.Log("📝 请填入腾讯云密钥后重启程序，否则将使用百度/本地裁剪方法");
                    return template;
                }

                var json = File.ReadAllText(ConfigPath);
                var config = JsonSerializer.Deserialize<AppConfig>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true,
                    WriteIndented = true
                }) ?? new AppConfig();

                // 启动时检查配置状态
                if (string.IsNullOrWhiteSpace(config.TencentSecretId) || string.IsNullOrWhiteSpace(config.TencentSecretKey))
                {
                    TencentOCR.Log("⚠️ 腾讯云配置为空，将使用百度/本地裁剪方法");
                }
                else
                {
                    TencentOCR.Log("✅ 已加载腾讯云配置（SecretId 前8位: " + config.TencentSecretId.Substring(0, Math.Min(8, config.TencentSecretId.Length)) + "...）");
                }

                return config;
            }
            catch (Exception ex)
            {
                TencentOCR.Log($"❌ 加载配置失败: {ex.Message}，使用默认配置");
                return new AppConfig();
            }
        }

        /// <summary>
        /// 保存配置到文件
        /// </summary>
        public void Save()
        {
            try
            {
                var json = JsonSerializer.Serialize(this, new JsonSerializerOptions
                {
                    WriteIndented = true,
                    Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
                });
                File.WriteAllText(ConfigPath, json);
                TencentOCR.Log($"💾 配置已保存到 {ConfigPath}");
            }
            catch (Exception ex)
            {
                TencentOCR.Log($"❌ 保存配置失败: {ex.Message}");
            }
        }

        /// <summary>
        /// 检查腾讯云配置是否可用
        /// </summary>
        public bool IsTencentConfigured()
        {
            return !string.IsNullOrWhiteSpace(TencentSecretId) && !string.IsNullOrWhiteSpace(TencentSecretKey);
        }
    }
}
