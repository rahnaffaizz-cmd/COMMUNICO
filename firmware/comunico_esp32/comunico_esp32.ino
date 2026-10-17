/*
  COMUNICO - ESP32-CAM (AI-Thinker) sebagai kamera, mikrofon, dan layar.

  ESP32 tidak membaca isyarat atau suara. Ia hanya mengirim gambar dan suara ke
  laptop lewat Wi-Fi, lalu menampilkan hasil yang dikirim balik oleh laptop.

    port 81  GET  /stream   video MJPEG (VGA 640x480)
    port 82  GET  /audio    suara mentah: 16 kHz, mono, 16 bit little-endian
    port 80  POST /pesan    teks hasil dari laptop, ditampilkan di OLED
                            ?dari=isyarat  (hasil bahasa isyarat)
                            ?dari=lawan    (hasil suara ke teks / ketikan)
    port 80  GET  /status   keadaan singkat (JSON)

  Di laptop:
    .\run.ps1 -Percakapan --esp32 <IP>    atau isi kolom ESP32 di kanan atas.
  IP-nya muncul di OLED dan Serial Monitor (115200) setelah tersambung Wi-Fi.

  RANGKAIAN
  ---------
  INMP441 (mikrofon I2S)       OLED SSD1306 128x64 (I2C)
    VDD  -> 3V3                  VCC -> 3V3
    GND  -> GND                  GND -> GND
    L/R  -> GND (kanal kiri)     SCL -> GPIO 14
    SCK  -> GPIO 12              SDA -> GPIO 15
    WS   -> GPIO 2
    SD   -> GPIO 13

  Catatan pin ESP32-CAM:
  - Slot kartu SD tidak boleh dipakai (pinnya dipakai mikrofon dan OLED).
  - GPIO 12 menentukan tegangan flash saat menyala. Jangan pasang pull-up di
    pin ini. Kalau ESP32 gagal menyala setelah mikrofon dipasang, lepas kabel
    SCK sebentar saat menekan RESET.
  - Pakai catu daya 5 V minimal 1 A. Kamera + Wi-Fi sering membuat ESP32-CAM
    restart sendiri ("Brownout detector") kalau dayanya kurang.

  LIBRARY (Arduino IDE -> Library Manager)
  ----------------------------------------
    Adafruit SSD1306, Adafruit GFX Library
  Board: "AI Thinker ESP32-CAM" (paket esp32 by Espressif, versi 2.x atau 3.x).
  Partition Scheme: "Huge APP (3MB No OTA)".
*/

#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "driver/i2s.h"

// ------------------------------------------------------------------ Wi-Fi
// Laptop dan ESP32 harus di jaringan yang sama (boleh hotspot HP/laptop).
const char *WIFI_NAMA  = "NAMA_WIFI";
const char *WIFI_SANDI = "SANDI_WIFI";

// ----------------------------------------------------------------- kamera
// Pin bawaan board AI-Thinker.
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

// Laptop sudah mencerminkan gambar seperti webcam. Ubah ini kalau gambar
// terbalik karena posisi modul kamera.
const bool KAMERA_BALIK_ATAS_BAWAH = false;
const bool KAMERA_CERMIN           = false;

// -------------------------------------------------------------- mikrofon
// Kamera memakai I2S0, jadi mikrofon harus di I2S1.
#define I2S_PORT     I2S_NUM_1
#define MIK_SCK      12
#define MIK_WS        2
#define MIK_SD       13
#define LAJU_SUARA   16000
// INMP441 mengirim 24 bit di dalam 32 bit. Digeser ke 16 bit; angka lebih kecil
// = lebih keras. 14 cocok untuk bicara biasa dari jarak sekitar 50 cm.
#define GESER_SUARA  14
// Kalau suara di laptop selalu hening, coba ganti ke I2S_CHANNEL_FMT_ONLY_RIGHT.
#define KANAL_MIK    I2S_CHANNEL_FMT_ONLY_LEFT

// ------------------------------------------------------------------ OLED
#define OLED_SDA     15
#define OLED_SCL     14
#define OLED_LEBAR  128
#define OLED_TINGGI  64
Adafruit_SSD1306 oled(OLED_LEBAR, OLED_TINGGI, &Wire, -1);
bool oledAda = false;

// Isi layar: 1 baris judul + 7 baris pesan, 21 huruf per baris (huruf ukuran 1).
#define BARIS_PESAN 7
#define LEBAR_BARIS 21
String barisLayar[BARIS_PESAN];
int jumlahBaris = 0;
String judulLayar = "COMUNICO";

void gambarLayar() {
  if (!oledAda) return;
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_WHITE);
  oled.setCursor(0, 0);
  oled.print(judulLayar);
  oled.drawFastHLine(0, 9, OLED_LEBAR, SSD1306_WHITE);
  for (int i = 0; i < jumlahBaris; i++) {
    oled.setCursor(0, 11 + i * 8);
    oled.print(barisLayar[i]);
  }
  oled.display();
}

void tambahBaris(const String &baris) {
  if (jumlahBaris == BARIS_PESAN) {           // layar penuh: geser ke atas
    for (int i = 1; i < BARIS_PESAN; i++) barisLayar[i - 1] = barisLayar[i];
    jumlahBaris--;
  }
  barisLayar[jumlahBaris++] = baris;
}

// Memecah pesan per kata supaya tidak terpotong di tengah kata.
void tampilkanPesan(const String &awalan, const String &teks) {
  String baris = awalan;
  int i = 0;
  while (i < (int)teks.length()) {
    int spasi = teks.indexOf(' ', i);
    if (spasi < 0) spasi = teks.length();
    String kata = teks.substring(i, spasi);
    i = spasi + 1;
    while (kata.length() > LEBAR_BARIS) {      // kata sangat panjang
      if (baris.length() > 0) { tambahBaris(baris); baris = ""; }
      tambahBaris(kata.substring(0, LEBAR_BARIS));
      kata = kata.substring(LEBAR_BARIS);
    }
    String calon = baris.length() == 0 ? kata
                 : (baris.endsWith(" ") ? baris + kata : baris + " " + kata);
    if (calon.length() > LEBAR_BARIS) {
      tambahBaris(baris);
      baris = kata;
    } else {
      baris = calon;
    }
  }
  if (baris.length() > 0) tambahBaris(baris);
  gambarLayar();
}

// --------------------------------------------------------------- video
#define BATAS "comunicoframe"
static const char *JENIS_ALIRAN = "multipart/x-mixed-replace;boundary=" BATAS;
static const char *KEPALA_BAGIAN =
    "\r\n--" BATAS "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static esp_err_t tanganiStream(httpd_req_t *req) {
  char kepala[96];
  esp_err_t hasil = httpd_resp_set_type(req, JENIS_ALIRAN);
  if (hasil != ESP_OK) return hasil;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { hasil = ESP_FAIL; break; }
    size_t n = snprintf(kepala, sizeof(kepala), KEPALA_BAGIAN, (unsigned)fb->len);
    hasil = httpd_resp_send_chunk(req, kepala, n);
    if (hasil == ESP_OK)
      hasil = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);
    if (hasil != ESP_OK) break;                // laptop menutup sambungan
  }
  return hasil;
}

// --------------------------------------------------------------- suara
static int32_t mentahSuara[512];
static int16_t pcmSuara[512];

static esp_err_t tanganiAudio(httpd_req_t *req) {
  httpd_resp_set_type(req, "application/octet-stream");
  i2s_zero_dma_buffer(I2S_PORT);
  while (true) {
    size_t dibaca = 0;
    if (i2s_read(I2S_PORT, mentahSuara, sizeof(mentahSuara), &dibaca,
                 portMAX_DELAY) != ESP_OK) break;
    int n = dibaca / sizeof(int32_t);
    for (int i = 0; i < n; i++) {
      int32_t s = mentahSuara[i] >> GESER_SUARA;
      if (s > 32767) s = 32767;
      if (s < -32768) s = -32768;
      pcmSuara[i] = (int16_t)s;
    }
    if (httpd_resp_send_chunk(req, (const char *)pcmSuara, n * sizeof(int16_t)) != ESP_OK)
      break;
  }
  return ESP_OK;
}

// --------------------------------------------------------------- pesan
static esp_err_t tanganiPesan(httpd_req_t *req) {
  char query[48] = {0};
  char dari[16] = "lawan";
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) == ESP_OK)
    httpd_query_key_value(query, "dari", dari, sizeof(dari));

  char isi[401];
  int panjang = min((int)req->content_len, (int)sizeof(isi) - 1);
  int diterima = 0;
  while (diterima < panjang) {
    int r = httpd_req_recv(req, isi + diterima, panjang - diterima);
    if (r <= 0) {
      if (r == HTTPD_SOCK_ERR_TIMEOUT) continue;
      return ESP_FAIL;
    }
    diterima += r;
  }
  isi[diterima] = '\0';

  String teks = String(isi);
  teks.trim();
  bool isyarat = strcmp(dari, "isyarat") == 0;
  Serial.printf("[%s] %s\n", isyarat ? "ISYARAT" : "LAWAN", teks.c_str());
  // '>' = hasil bahasa isyarat, '<' = ucapan/ketikan lawan bicara.
  tampilkanPesan(isyarat ? "> " : "< ", teks);

  httpd_resp_set_type(req, "text/plain");
  return httpd_resp_sendstr(req, "ok");
}

static esp_err_t tanganiStatus(httpd_req_t *req) {
  char isi[160];
  snprintf(isi, sizeof(isi),
           "{\"nama\":\"COMUNICO-ESP32\",\"ip\":\"%s\",\"rssi\":%d,\"oled\":%s}",
           WiFi.localIP().toString().c_str(), WiFi.RSSI(), oledAda ? "true" : "false");
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_sendstr(req, isi);
}

// Tiap aliran punya server sendiri, karena satu server hanya mengerjakan
// satu permintaan sekaligus dan aliran video/suara tidak pernah selesai.
httpd_handle_t mulaiServer(uint16_t port) {
  httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
  cfg.server_port = port;
  cfg.ctrl_port = 32768 + port;
  cfg.lru_purge_enable = true;                 // laptop menyambung ulang: buang yang lama
  cfg.max_open_sockets = 3;
  httpd_handle_t server = NULL;
  if (httpd_start(&server, &cfg) != ESP_OK) {
    Serial.printf("Server port %u gagal dimulai\n", port);
    return NULL;
  }
  return server;
}

void daftar(httpd_handle_t server, const char *uri, httpd_method_t metode,
            esp_err_t (*penangan)(httpd_req_t *)) {
  if (!server) return;
  httpd_uri_t u = {};
  u.uri = uri;
  u.method = metode;
  u.handler = penangan;
  httpd_register_uri_handler(server, &u);
}

// --------------------------------------------------------------- awal
bool mulaiKamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  // VGA cukup untuk MediaPipe membaca tangan; QVGA lebih lancar tapi kurang teliti.
  c.frame_size = FRAMESIZE_VGA;
  c.jpeg_quality = 12;                         // 10-15: kecil = lebih tajam, lebih berat
  c.fb_count = psramFound() ? 2 : 1;
  c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;            // selalu gambar terbaru, bukan antrean

  if (esp_camera_init(&c) != ESP_OK) return false;
  sensor_t *s = esp_camera_sensor_get();
  s->set_vflip(s, KAMERA_BALIK_ATAS_BAWAH ? 1 : 0);
  s->set_hmirror(s, KAMERA_CERMIN ? 1 : 0);
  return true;
}

bool mulaiMikrofon() {
  i2s_config_t cfg = {};
  cfg.mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX);
  cfg.sample_rate = LAJU_SUARA;
  cfg.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  cfg.channel_format = KANAL_MIK;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  cfg.dma_buf_count = 8;
  cfg.dma_buf_len = 256;
  cfg.use_apll = false;

  i2s_pin_config_t pin = {};
  pin.mck_io_num = I2S_PIN_NO_CHANGE;
  pin.bck_io_num = MIK_SCK;
  pin.ws_io_num = MIK_WS;
  pin.data_out_num = I2S_PIN_NO_CHANGE;
  pin.data_in_num = MIK_SD;

  if (i2s_driver_install(I2S_PORT, &cfg, 0, NULL) != ESP_OK) return false;
  return i2s_set_pin(I2S_PORT, &pin) == ESP_OK;
}

void setup() {
  Serial.begin(115200);
  Serial.println("\nCOMUNICO ESP32-CAM");

  Wire.begin(OLED_SDA, OLED_SCL);
  oledAda = oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  if (!oledAda) Serial.println("OLED tidak ditemukan (alamat 0x3C?)");
  judulLayar = "COMUNICO";
  tampilkanPesan("", "Menyiapkan kamera...");

  if (!mulaiKamera()) {
    Serial.println("Kamera gagal dimulai");
    tampilkanPesan("", "KAMERA GAGAL. Cek kabel pita kamera lalu RESET.");
    while (true) delay(1000);
  }
  if (!mulaiMikrofon()) {
    Serial.println("Mikrofon gagal dimulai");
    tampilkanPesan("", "Mikrofon gagal; video tetap jalan.");
  }

  tampilkanPesan("", String("Wi-Fi: ") + WIFI_NAMA);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);                        // hemat daya membuat video tersendat
  WiFi.begin(WIFI_NAMA, WIFI_SANDI);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  String ip = WiFi.localIP().toString();
  Serial.printf("\nTersambung. Isi kolom ESP32 di laptop dengan: %s\n", ip.c_str());

  httpd_handle_t serverPesan = mulaiServer(80);
  daftar(serverPesan, "/pesan", HTTP_POST, tanganiPesan);
  daftar(serverPesan, "/status", HTTP_GET, tanganiStatus);
  daftar(mulaiServer(81), "/stream", HTTP_GET, tanganiStream);
  daftar(mulaiServer(82), "/audio", HTTP_GET, tanganiAudio);

  judulLayar = "IP " + ip;
  jumlahBaris = 0;
  tampilkanPesan("", "Siap. Isi IP ini di kolom ESP32 laptop.");
}

void loop() {
  // Semua pekerjaan ada di server HTTP. Di sini hanya menjaga Wi-Fi.
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.reconnect();
    delay(2000);
  }
  delay(100);
}
