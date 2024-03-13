import scrapy

class KitapSpider(scrapy.Spider):
    name = 'kitap_spider3'

    def start_requests(self):
        lst = [f"https://www.kitapsec.com/Cok-Satanlar//{i}-6-0a0-0-0-0-0-0.xhtml" for i in range(1, 4254)]
        for url in lst:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for kitap in response.css('div.Ks_UrunSatir'):
            kitap_ismi_parts = kitap.css('span[itemprop="name"]::text').getall()

            yazar_ismi = kitap_ismi_parts[0] if kitap_ismi_parts else 'Unknown'
            kitap_ismi = kitap_ismi_parts[1] if len(kitap_ismi_parts) > 1 else 'Unknown'
            yayinevi = kitap_ismi_parts[2] if len(kitap_ismi_parts) > 2 else 'Unknown'

            # Exclude yazar_ismi from kitap_ismi if included
            if yazar_ismi in kitap_ismi:
                kitap_ismi = kitap_ismi.replace(yazar_ismi, '').strip()

            # Exclude yayinevi from yazar_ismi if included
            if yayinevi in kitap_ismi:
                kitap_ismi = kitap_ismi.replace(yayinevi, '').strip()

            yield {
                'yazar_ismi': yazar_ismi,
                'kitap_ismi': kitap_ismi,
                'yayinevi': yayinevi,
                'piyasa_fiyati': kitap.css('font.piyasa::text').get(),
                'satis_fiyati': kitap.css('font.satis::text').get(),
            }
