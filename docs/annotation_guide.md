# Etiketleme Rehberi

> Placeholder — etiketleme kuralları MVP-2 başlangıcında kesinleştirilecek.

## Genel Prensipler

- Bounding box'lar nesnenin tüm görünür kısmını kapsamalı, fazla padding olmamalıdır.
- Kısmen örtülmüş (occluded) ama tanınabilir nesneler etiketlenir; örtüsü %70'ten fazla olanlar atlanır.
- Hasarlı izolatör için ayrı sınıf (`damaged_insulator`) kullanılır.

## Önerilen Araçlar

- [Roboflow](https://roboflow.com/) — buluttan etiketleme + augmentation + YOLO export.
- [Label Studio](https://labelstud.io/) — self-host.
- [CVAT](https://www.cvat.ai/) — video / 3D destekli.

## Export

YOLO format (txt + images). Class mapping `config/model_config.yaml`
içindeki `classes` listesi ile birebir eşleşmelidir.
