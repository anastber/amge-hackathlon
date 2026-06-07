"""
=============================================================
  BUSINESS INSIGHTS EXTRACTOR — E-Commerce Marketplace
  Pour la présentation stratégique (audience non-technique)
=============================================================
Lance avec : python business_insights_extractor.py
"""

from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 0. CHARGEMENT — adapte les chemins si besoin
# ─────────────────────────────────────────────
# Si tu as train + test séparés, charge les deux et concatène
# df = pd.concat([pd.read_csv("train.csv"), pd.read_csv("test.csv")], ignore_index=True)
df = pd.read_csv("dataset/train.csv")  # <-- adapte le nom du fichier

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Lignes        : {len(df):,}")
print(f"Commandes uniques : {df['order_id'].nunique():,}")
print(f"Clients uniques   : {df['customer_unique_id'].nunique():,}")
print(f"Vendeurs uniques  : {df['seller_id'].nunique():,}")
print(f"Produits uniques  : {df['product_id'].nunique():,}")
print(f"Catégories        : {df['product_category_name_english'].nunique()}")
print(
    f"Période           : {df['order_purchase_timestamp'].min()} → {df['order_purchase_timestamp'].max()}"
)

# ─────────────────────────────────────────────
# 1. DÉLAIS DE LIVRAISON (variable cible)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. DÉLAIS DE LIVRAISON (delivery_time_days)")
print("=" * 60)
dt = df["delivery_time_days"]
print(f"Moyenne   : {dt.mean():.1f} jours")
print(f"Médiane   : {dt.median():.1f} jours")
print(f"P25       : {dt.quantile(0.25):.1f} jours")
print(f"P75       : {dt.quantile(0.75):.1f} jours")
print(f"P90       : {dt.quantile(0.90):.1f} jours")
print(f"P95       : {dt.quantile(0.95):.1f} jours")
print(f"Max       : {dt.max():.1f} jours")

# Seuils satisfaction
seuils = [7, 14, 21, 30]
for s in seuils:
    pct = (dt <= s).mean() * 100
    print(f"  → {pct:.1f}% des commandes livrées en ≤ {s} jours")

# ─────────────────────────────────────────────
# 2. CONCENTRATION DES VENDEURS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CONCENTRATION DES VENDEURS")
print("=" * 60)

# Revenu par vendeur
df["revenue"] = df["price"] * df["quantity"]
seller_rev = df.groupby("seller_id")["revenue"].sum().sort_values(ascending=False)
total_rev = seller_rev.sum()

top10_pct = seller_rev.head(10).sum() / total_rev * 100
top50_pct = seller_rev.head(50).sum() / total_rev * 100
top100_pct = seller_rev.head(100).sum() / total_rev * 100
nb_sellers = len(seller_rev)

print(f"Nombre total de vendeurs : {nb_sellers:,}")
print(f"Top 10  vendeurs → {top10_pct:.1f}% du CA")
print(f"Top 50  vendeurs → {top50_pct:.1f}% du CA")
print(f"Top 100 vendeurs → {top100_pct:.1f}% du CA")
print(
    f"Bottom 50% vendeurs → {seller_rev.tail(nb_sellers//2).sum()/total_rev*100:.1f}% du CA"
)

# Commandes par vendeur
seller_orders = (
    df.groupby("seller_id")["order_id"].nunique().sort_values(ascending=False)
)
print(f"\nCommandes/vendeur — médiane : {seller_orders.median():.0f}")
print(f"Commandes/vendeur — top 10% : {seller_orders.quantile(0.90):.0f}+")
print(
    f"Vendeurs avec < 5 commandes  : {(seller_orders < 5).sum():,} ({(seller_orders < 5).mean()*100:.1f}%)"
)
print(
    f"Vendeurs avec < 10 commandes : {(seller_orders < 10).sum():,} ({(seller_orders < 10).mean()*100:.1f}%)"
)

# ─────────────────────────────────────────────
# 3. ANALYSE DES PRIX & FRAIS DE PORT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. PRIX & FRAIS DE PORT")
print("=" * 60)
print(f"Prix moyen produit   : {df['price'].mean():.2f} BRL")
print(f"Prix médian produit  : {df['price'].median():.2f} BRL")
print(f"Freight moyen        : {df['freight_value'].mean():.2f} BRL")
print(f"Freight médian       : {df['freight_value'].median():.2f} BRL")

df["freight_ratio"] = df["freight_value"] / (df["price"] * df["quantity"])
print(f"\nRatio freight/prix moyen  : {df['freight_ratio'].mean()*100:.1f}%")
print(f"Ratio freight/prix médian : {df['freight_ratio'].median()*100:.1f}%")
print(
    f"Commandes où freight > prix : {(df['freight_value'] > df['price']).mean()*100:.1f}%"
)

# ─────────────────────────────────────────────
# 4. TOP CATÉGORIES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. TOP CATÉGORIES (volume & CA)")
print("=" * 60)
cat_stats = (
    df.groupby("product_category_name_english")
    .agg(
        nb_commandes=("order_id", "nunique"),
        ca_total=("revenue", "sum"),
        delai_moyen=("delivery_time_days", "mean"),
        prix_moyen=("price", "mean"),
    )
    .sort_values("nb_commandes", ascending=False)
)

print("\nTop 15 catégories par volume de commandes :")
print(cat_stats.head(15).to_string())

print("\nTop 10 catégories par CA :")
print(
    cat_stats.sort_values("ca_total", ascending=False)
    .head(10)[["ca_total", "delai_moyen"]]
    .to_string()
)

print("\nTop 10 catégories avec les délais les PLUS LONGS :")
print(
    cat_stats[cat_stats["nb_commandes"] > 50]
    .sort_values("delai_moyen", ascending=False)
    .head(10)[["nb_commandes", "delai_moyen"]]
    .to_string()
)

print("\nTop 10 catégories avec les délais les PLUS COURTS :")
print(
    cat_stats[cat_stats["nb_commandes"] > 50]
    .sort_values("delai_moyen")
    .head(10)[["nb_commandes", "delai_moyen"]]
    .to_string()
)

# ─────────────────────────────────────────────
# 5. GÉOGRAPHIE — DISTANCE & DÉLAIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. GÉOGRAPHIE — IMPACT DE LA DISTANCE")
print("=" * 60)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


geo_df = df.dropna(
    subset=["customer_lat", "customer_lng", "seller_lat", "seller_lng"]
).copy()
geo_df["distance_km"] = geo_df.apply(
    lambda r: haversine(
        r["customer_lat"], r["customer_lng"], r["seller_lat"], r["seller_lng"]
    ),
    axis=1,
)

print(f"Distance moyenne client-vendeur : {geo_df['distance_km'].mean():.0f} km")
print(f"Distance médiane                : {geo_df['distance_km'].median():.0f} km")

# Quartiles de distance vs délai
geo_df["dist_quartile"] = pd.qcut(
    geo_df["distance_km"], q=4, labels=["Q1 (proche)", "Q2", "Q3", "Q4 (loin)"]
)
dist_delay = geo_df.groupby("dist_quartile")["delivery_time_days"].mean()
print("\nDélai moyen par quartile de distance :")
print(dist_delay.to_string())

# Top villes clientes
print("\nTop 10 villes clientes :")
print(df["customer_city"].value_counts().head(10).to_string())

# Top villes vendeuses
print("\nTop 10 villes vendeuses :")
print(df["seller_city"].value_counts().head(10).to_string())

# ─────────────────────────────────────────────
# 6. QUALITÉ DES LISTINGS VENDEURS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. QUALITÉ DES LISTINGS (photos, description)")
print("=" * 60)

listing = df[
    [
        "product_photos_qty",
        "product_description_length",
        "product_name_length",
        "delivery_time_days",
        "revenue",
    ]
].dropna()

print(f"Photos/produit — moyenne  : {listing['product_photos_qty'].mean():.1f}")
print(f"Photos/produit — médiane  : {listing['product_photos_qty'].median():.0f}")
print(
    f"Produits sans photos      : {(listing['product_photos_qty'] == 0).mean()*100:.1f}%"
)
print(
    f"Produits avec 1 photo     : {(listing['product_photos_qty'] == 1).mean()*100:.1f}%"
)
print(
    f"Produits avec 3+ photos   : {(listing['product_photos_qty'] >= 3).mean()*100:.1f}%"
)

# Photos vs CA
listing["photo_bin"] = pd.cut(
    listing["product_photos_qty"], bins=[0, 1, 2, 3, 20], labels=["0-1", "2", "3", "4+"]
)
print("\nCA moyen par nombre de photos :")
print(listing.groupby("photo_bin")["revenue"].mean().to_string())

print(
    f"\nDescription — moyenne  : {listing['product_description_length'].mean():.0f} chars"
)
print(
    f"Description — médiane  : {listing['product_description_length'].median():.0f} chars"
)

# ─────────────────────────────────────────────
# 7. TEMPORALITÉ — SAISONNALITÉ
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. SAISONNALITÉ DES COMMANDES")
print("=" * 60)
df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
df["month"] = df["order_purchase_timestamp"].dt.month
df["hour"] = df["order_purchase_timestamp"].dt.hour
df["dow"] = df["order_purchase_timestamp"].dt.dayofweek  # 0=lundi

monthly = df.groupby("month").agg(
    nb_commandes=("order_id", "nunique"),
    ca=("revenue", "sum"),
    delai_moyen=("delivery_time_days", "mean"),
)
print("\nCommandes & délai par mois :")
print(monthly.to_string())

print("\nCommandes par heure de la journée (top 5) :")
print(
    df.groupby("hour")["order_id"]
    .count()
    .sort_values(ascending=False)
    .head(5)
    .to_string()
)

print("\nCommandes par jour de la semaine :")
dow_labels = {
    0: "Lundi",
    1: "Mardi",
    2: "Mercredi",
    3: "Jeudi",
    4: "Vendredi",
    5: "Samedi",
    6: "Dimanche",
}
dow_counts = df.groupby("dow")["order_id"].count().rename(dow_labels)
print(dow_counts.to_string())

# ─────────────────────────────────────────────
# 8. RÉCURRENCE CLIENTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. RÉCURRENCE / FIDÉLISATION CLIENTS")
print("=" * 60)
customer_orders = df.groupby("customer_unique_id")["order_id"].nunique()
print(
    f"Clients avec 1 seule commande  : {(customer_orders == 1).sum():,} ({(customer_orders == 1).mean()*100:.1f}%)"
)
print(
    f"Clients avec 2 commandes       : {(customer_orders == 2).sum():,} ({(customer_orders == 2).mean()*100:.1f}%)"
)
print(
    f"Clients avec 3+ commandes      : {(customer_orders >= 3).sum():,} ({(customer_orders >= 3).mean()*100:.1f}%)"
)
print(
    f"Clients avec 5+ commandes      : {(customer_orders >= 5).sum():,} ({(customer_orders >= 5).mean()*100:.1f}%)"
)
print(f"Commandes/client — médiane     : {customer_orders.median():.0f}")
print(f"Commandes/client — moyenne     : {customer_orders.mean():.2f}")

# ─────────────────────────────────────────────
# 9. RÉSUMÉ EXÉCUTIF — CHIFFRES CLÉS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. RÉSUMÉ EXÉCUTIF — CHIFFRES CLÉS POUR LA PRÉSENTATION")
print("=" * 60)
print(f"📦 Volume total commandes     : {df['order_id'].nunique():,}")
print(f"👥 Base clients               : {df['customer_unique_id'].nunique():,}")
print(f"🏪 Base vendeurs              : {df['seller_id'].nunique():,}")
print(f"💰 CA total (BRL)             : {total_rev:,.0f}")
print(f"⏱  Délai moyen livraison      : {df['delivery_time_days'].mean():.1f} jours")
print(f"📍 Distance moyenne           : {geo_df['distance_km'].mean():.0f} km")
print(f"🔄 Taux clients récurrents    : {(customer_orders >= 2).mean()*100:.1f}%")
print(f"🎯 Top 10 vendeurs = X% du CA (voir section 2)")
print(
    f"📸 % produits avec ≥3 photos  : {(listing['product_photos_qty'] >= 3).mean()*100:.1f}%"
)

print("\n✅ Script terminé. Copie-colle les outputs ici pour la présentation.")
