# Dataset Rehberi

> Placeholder — detaylı dataset hazırlama akışı MVP-2 fazında doldurulacak.

## Public Dataset'ler

| Kısaltma   | Açıklama                                        |
|------------|-------------------------------------------------|
| INSPLAD    | Insulator inspection (defect classification)     |
| MPID       | Multi-class powerline insulator dataset          |
| STN-PLAD   | Power line asset detection                       |
| UPI        | Unified Powerline Insulator dataset              |
| CPLID      | Chinese Powerline Insulator dataset              |
| TTPLA      | Transmission Tower & Power Line Aerial dataset   |

Her birinin lisans şartları farklıdır; indirip kullanmadan önce
`scripts/download_datasets.py` içinde sağlanan dokümantasyon linklerini
gözden geçiriniz.

## YOLO Format Dönüşümü

`scripts/prepare_yolo_dataset.py` ile `data/external/` altındaki kaynaklar
`data/yolo/{train,val,test}` altına YOLO formatında (images + labels)
yazılır. Sınıf eşlemesi `config/model_config.yaml` ile senkronize tutulur.
