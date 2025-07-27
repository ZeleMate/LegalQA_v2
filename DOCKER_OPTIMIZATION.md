# Docker Optimalizációk

Ez a dokumentum leírja a LegalQA projekt Docker optimalizációit a CI/CD folyamatok gyorsítására.

## Dockerfile Változatok

### 1. Dockerfile (Eredeti)
- **Cél**: Teljes fejlesztői környezet
- **Függőségek**: Minden szükséges csomag
- **Build idő**: ~150 másodperc
- **Használat**: Lokális fejlesztés, production

### 2. Dockerfile.ci (CI Optimalizált)
- **Cél**: CI/CD környezet
- **Függőségek**: Teljes függőségek, de jobb cache stratégiával
- **Build idő**: ~110 másodperc
- **Használat**: GitHub Actions CI

### 3. Dockerfile.minimal (Minimális)
- **Cél**: Gyors CI tesztelés
- **Függőségek**: Csak a legszükségesebb csomagok
- **Build idő**: ~50 másodperc
- **Használat**: GitHub Actions CI (jelenlegi)

## Optimalizációs Stratégiák

### 1. Réteges Cache
- Függőségek külön rétegekben telepítése
- Kisebb csomagok először
- Nagyobb csomagok utoljára

### 2. .dockerignore Optimalizáció
- Felesleges fájlok kizárása
- Build context méretének csökkentése

### 3. Multi-stage Build
- Builder és production stage elkülönítése
- Csak szükséges fájlok másolása

### 4. GitHub Actions Cache
- Registry cache használata
- Layer cache optimalizálás

## Használat

### Lokális Fejlesztés
```bash
docker build -t legalqa:dev .
```

### CI Tesztelés
```bash
docker build -f Dockerfile.minimal -t legalqa:ci .
```

### Production Build
```bash
docker build -t legalqa:prod .
```

## Teljesítmény Összehasonlítás

| Dockerfile | Build Idő | Méret | Használat |
|------------|-----------|-------|-----------|
| Eredeti | ~150s | Nagy | Dev/Prod |
| CI | ~110s | Közepes | CI |
| Minimális | ~50s | Kicsi | CI |

## Jövőbeli Optimalizációk

1. **Alpine Linux**: Kisebb base image
2. **Pre-built Wheels**: Előfordított csomagok használata
3. **Distroless**: Minimális runtime image
4. **Build Cache**: Jobb cache stratégiák

## Hibaelhárítás

### Build Időtúllépés
- Használja a `Dockerfile.minimal`-t CI-hez
- Növelje a timeout-ot a workflow-ban
- Ellenőrizze a cache beállításokat

### Cache Problémák
- Törölje a régi cache-eket
- Frissítse a cache kulcsokat
- Használjon registry cache-t 