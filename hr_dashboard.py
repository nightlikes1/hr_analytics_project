import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import shap

# Yeni Modüler Yapı Importları
from src.database.engine import get_db_engine, load_employees_from_db, load_single_employee
from src.models.predictor import load_model_resources, preprocess_input
from src.reports.pdf_generator import generate_pdf_report
from src.utils.hr_math import calculate_costs, calculate_roi

# Sayfa Konfigürasyonu
st.set_page_config(page_title="HR Analytics Platform", layout="wide", page_icon="📊")

# Tasarımı sadeleştir
st.title("📊 HR Analytics Platformu")
st.markdown("---")

# -------------------------------------------------
# 1. Kaynak Yükleme ve Ayarlar
# -------------------------------------------------
@st.cache_resource
def init_app():
    model, feature_names, model_path = load_model_resources()
    engine = get_db_engine()
    return model, feature_names, engine, model_path

try:
    model, feature_names, engine, active_model_path = init_app()
except Exception as e:
    st.error(f"Kaynaklar yüklenirken hata: {e}")
    st.stop()

# -------------------------------------------------
# 2. Sidebar
# -------------------------------------------------
st.sidebar.title("🚀 HR Analytics")
st.sidebar.info(f"Aktif Model: {os.path.basename(active_model_path)}")

st.sidebar.subheader("⚙️ Analiz Ayarları")
analiz_derinligi = st.sidebar.select_slider("Analiz Detay Seviyesi", options=["Temel", "Orta", "Gelişmiş"], value="Gelişmiş")
st.sidebar.success("✅ Yerel Analiz Motoru Aktif")

menu = st.sidebar.radio("Ana Menü", [
    "🏠 Karşılama & Uyarılar",
    "📁 Veri Portalı", 
    "🏥 Departman Analizi", 
    "📊 9-Box Yetenek Matrisi",
    "🔮 Tahmin & What-If", 
    "💰 Müdahale & ROI Analizi",
    "👯 Çalışan Kıyaslama",
    "🤖 Strateji Uzmanı", 
    "🔍 Model Şeffaflığı"
])

# -------------------------------------------------
# 4. Sayfa: Karşılama & Uyarılar
# -------------------------------------------------
if menu == "🏠 Karşılama & Uyarılar":
    st.header("🏠 İK Yönetim Bildirim Merkezi")
    df_db = load_employees_from_db(engine)
    processed_all = preprocess_input(df_db, feature_names)
    df_db['Risk'] = model.predict_proba(processed_all)[:, 1]
    
    critical_cases = df_db[df_db['Risk'] > 0.7].sort_values('Risk', ascending=False)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚨 Kritik Ayrılma Uyarısı")
        if not critical_cases.empty:
            for _, r in critical_cases.head(5).iterrows():
                st.error(f"**KRİTİK RİSK:** Çalışan #{r['EmployeeNumber']} ({r['JobRole']})  \n"
                         f"**Risk Skoru: %{r['Risk']*100:.1f}** | Tahmini Kayıp: ${calculate_costs(r['Risk'], r['MonthlyIncome']):,.0f}")
        else:
            st.success("Her şey yolunda! Şu an için kritik risk tespit edilmedi.")

    with col2:
        st.subheader("📊 Hızlı Özet")
        st.metric("Toplam Çalışan", len(df_db))
        st.metric("Yüksek Risk (>%50)", len(df_db[df_db['Risk'] > 0.5]))
        st.metric("Toplam Potansiyel Kayıp", f"${df_db.apply(lambda x: calculate_costs(x['Risk'], x['MonthlyIncome']), axis=1).sum():,.0f}")

# -------------------------------------------------
# 5. Sayfa: Veri Portalı
# -------------------------------------------------
elif menu == "📁 Veri Portalı":
    st.header("📁 Veri Yükleme ve Yönetim Portalı")
    uploaded_file = st.file_uploader("Çalışan Listesi Yükle (CSV)", type="csv")
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        st.success(f"{len(df_new)} yeni kayıt yüklendi.")
        if st.checkbox("Tahminleri Çalıştır"):
            with st.spinner("Analiz ediliyor..."):
                processed = preprocess_input(df_new, feature_names)
                probs = model.predict_proba(processed)[:, 1]
                df_new['Ayrılma_Riski'] = np.round(probs * 100, 2)
                st.dataframe(df_new[['EmployeeNumber', 'JobRole', 'Department', 'Ayrılma_Riski']].sort_values('Ayrılma_Riski', ascending=False))
                csv = df_new.to_csv(index=False).encode('utf-8')
                st.download_button("Tahminleri İndir (CSV)", csv, "hr_prediction_results.csv", "text/csv")

# -------------------------------------------------
# 6. Sayfa: Departman Analizi
# -------------------------------------------------
elif menu == "🏥 Departman Analizi":
    st.header("🏥 Departman & Rol Bazlı Derinlemesine Risk Analizi")
    st.markdown("*Departmanların risk profillerini çok boyutlu olarak inceleyin ve stratejik kararlar alın*")
    df_db = load_employees_from_db(engine)
    processed_all = preprocess_input(df_db, feature_names)
    df_db['Risk_Skoru'] = model.predict_proba(processed_all)[:, 1]
    df_db['Risk_Kategori'] = pd.cut(df_db['Risk_Skoru'], bins=[0, 0.3, 0.6, 1.0], labels=['Düşük', 'Orta', 'Yüksek'])

    # ── Özet Metrikler ──
    dept_stats = df_db.groupby('Department', observed=False).agg(
        Ort_Risk=('Risk_Skoru', 'mean'), Toplam=('Risk_Skoru', 'count'),
        Yüksek_Risk=('Risk_Skoru', lambda x: (x > 0.5).sum()),
        Ort_Maaş=('MonthlyIncome', 'mean'),
        Ort_Kıdem=('YearsAtCompany', 'mean'),
        Ort_Memnuniyet=('JobSatisfaction', 'mean')
    ).reset_index()
    dept_stats['Risk_Oranı'] = (dept_stats['Yüksek_Risk'] / dept_stats['Toplam'] * 100).round(1)
    dept_stats['Potansiyel_Kayıp'] = dept_stats.apply(
        lambda r: df_db[df_db['Department'] == r['Department']].apply(
            lambda x: calculate_costs(x['Risk_Skoru'], x['MonthlyIncome']), axis=1).sum(), axis=1)

    m1, m2, m3 = st.columns(3)
    riskiest = dept_stats.loc[dept_stats['Ort_Risk'].idxmax()]
    safest = dept_stats.loc[dept_stats['Ort_Risk'].idxmin()]
    m1.metric("🔴 En Riskli Departman", riskiest['Department'], f"Ort. Risk: %{riskiest['Ort_Risk']*100:.1f}")
    m2.metric("🟢 En Güvenli Departman", safest['Department'], f"Ort. Risk: %{safest['Ort_Risk']*100:.1f}")
    m3.metric("💸 Toplam Potansiyel Kayıp", f"${dept_stats['Potansiyel_Kayıp'].sum():,.0f}")
    st.markdown("---")

    # ── Departman Risk Bar + Rol Risk Bar ──
    st.subheader("📊 Departman ve Rol Risk Haritası")
    col1, col2 = st.columns(2)
    with col1:
        fig_dept = px.bar(dept_stats.sort_values('Ort_Risk', ascending=False), x='Department', y='Ort_Risk',
                          color='Ort_Risk', color_continuous_scale='RdYlGn_r', title="Departman Ortalama Risk",
                          text=dept_stats.sort_values('Ort_Risk', ascending=False)['Ort_Risk'].apply(lambda x: f'%{x*100:.1f}'))
        fig_dept.update_traces(textposition='outside')
        fig_dept.update_layout(yaxis_title="Ortalama Risk Skoru", xaxis_title="")
        st.plotly_chart(fig_dept, use_container_width=True)
    with col2:
        role_risk = df_db.groupby('JobRole', observed=False)['Risk_Skoru'].mean().sort_values(ascending=True).reset_index()
        fig_role = px.bar(role_risk, x='Risk_Skoru', y='JobRole', orientation='h', color='Risk_Skoru',
                          color_continuous_scale='RdYlGn_r', title="İş Rolüne Göre Risk Sıralaması")
        fig_role.update_layout(yaxis_title="", xaxis_title="Ortalama Risk Skoru")
        st.plotly_chart(fig_role, use_container_width=True)

    st.markdown("---")

    # ── Departman × Rol Heatmap ──
    st.subheader("🗺️ Departman × Rol Çapraz Risk Haritası")
    heatmap_data = df_db.pivot_table(index='JobRole', columns='Department', values='Risk_Skoru', aggfunc='mean')
    fig_heat = px.imshow(heatmap_data, text_auto='.2f', color_continuous_scale='RdYlGn_r',
                         title="Departman ve Rol Bazında Ortalama Risk (Heatmap)", aspect='auto')
    fig_heat.update_layout(height=450, xaxis_title="Departman", yaxis_title="İş Rolü")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Treemap: Departman > Rol > Risk ──
    st.subheader("🌳 Hiyerarşik Risk Ağacı (Treemap)")
    tree_data = df_db.groupby(['Department', 'JobRole'], observed=False).agg(
        Kişi=('EmployeeNumber', 'count'), Ort_Risk=('Risk_Skoru', 'mean')).reset_index()
    fig_tree = px.treemap(tree_data, path=['Department', 'JobRole'], values='Kişi', color='Ort_Risk',
                          color_continuous_scale='RdYlGn_r', title="Departman > Rol Hiyerarşisi (Büyüklük: Kişi, Renk: Risk)")
    fig_tree.update_layout(height=500)
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("---")

    # ── Maaş vs Risk Scatter (Departmana göre) ──
    st.subheader("💰 Maaş vs Risk İlişkisi (Departman Bazlı)")
    scat_col1, scat_col2 = st.columns([3, 1])
    with scat_col1:
        fig_scat = px.scatter(df_db, x='MonthlyIncome', y='Risk_Skoru', color='Department',
                              size='YearsAtCompany', hover_data=['EmployeeNumber', 'JobRole'],
                              title="Her Nokta Bir Çalışan (Büyüklük: Kıdem Yılı)", opacity=0.7)
        fig_scat.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Yüksek Risk Eşiği")
        fig_scat.update_layout(xaxis_title="Aylık Maaş ($)", yaxis_title="Ayrılma Riski", height=450)
        st.plotly_chart(fig_scat, use_container_width=True)
    with scat_col2:
        st.markdown("##### 📋 Yorumlama")
        st.markdown("- **Sol üst:** Düşük maaş + Yüksek risk → 🔴 Acil müdahale")
        st.markdown("- **Sağ alt:** Yüksek maaş + Düşük risk → 🟢 Güvenli")
        st.markdown("- **Büyük noktalar:** Kıdemli çalışanlar")
        high_risk_low_pay = df_db[(df_db['Risk_Skoru'] > 0.5) & (df_db['MonthlyIncome'] < df_db['MonthlyIncome'].median())]
        st.metric("🔴 Düşük Maaş + Yüksek Risk", f"{len(high_risk_low_pay)} kişi")

    st.markdown("---")

    # ── Departman Detay Tablosu ──
    st.subheader("📋 Departman Karşılaştırma Tablosu")
    display_stats = dept_stats.rename(columns={
        'Department': 'Departman', 'Ort_Risk': 'Ort. Risk', 'Toplam': 'Çalışan Sayısı',
        'Yüksek_Risk': 'Yüksek Riskli', 'Ort_Maaş': 'Ort. Maaş ($)', 'Ort_Kıdem': 'Ort. Kıdem (Yıl)',
        'Ort_Memnuniyet': 'Ort. Memnuniyet', 'Risk_Oranı': 'Risk Oranı (%)', 'Potansiyel_Kayıp': 'Pot. Kayıp ($)'
    })
    st.dataframe(display_stats.style.background_gradient(subset=['Ort. Risk'], cmap='RdYlGn_r')
                 .format({'Ort. Risk': '{:.3f}', 'Ort. Maaş ($)': '${:,.0f}', 'Ort. Kıdem (Yıl)': '{:.1f}',
                          'Ort. Memnuniyet': '{:.2f}', 'Pot. Kayıp ($)': '${:,.0f}'}), use_container_width=True)

    # ── Aksiyon Önerileri ──
    st.markdown("---")
    st.subheader("💡 Departman Bazlı Aksiyon Önerileri")
    for _, dept in dept_stats.sort_values('Ort_Risk', ascending=False).iterrows():
        severity = "🔴" if dept['Ort_Risk'] > 0.35 else "🟡" if dept['Ort_Risk'] > 0.25 else "🟢"
        with st.expander(f"{severity} {dept['Department']} — Risk: %{dept['Ort_Risk']*100:.1f} | {int(dept['Yüksek_Risk'])} yüksek riskli"):
            ac1, ac2 = st.columns(2)
            with ac1:
                st.metric("Çalışan Sayısı", int(dept['Toplam']))
                st.metric("Ortalama Maaş", f"${dept['Ort_Maaş']:,.0f}")
                st.metric("Potansiyel Kayıp", f"${dept['Potansiyel_Kayıp']:,.0f}")
            with ac2:
                st.metric("Ortalama Kıdem", f"{dept['Ort_Kıdem']:.1f} yıl")
                st.metric("Memnuniyet", f"{dept['Ort_Memnuniyet']:.2f}/4")
                st.metric("Risk Oranı", f"%{dept['Risk_Oranı']:.1f}")
            if dept['Ort_Risk'] > 0.35:
                st.error("⚠️ **Acil:** Birebir görüşmeler, maaş revizyonu ve esnek çalışma modeli uygulanmalı.")
            elif dept['Ort_Risk'] > 0.25:
                st.warning("⚡ **Önemli:** Memnuniyet anketi, kariyer gelişim planı ve takım etkinlikleri düzenlenmeli.")
            else:
                st.success("✅ Risk kontrol altında. Mevcut koşullar korunmalı ve düzenli izleme sürdürülmeli.")

# -------------------------------------------------
# 7. Sayfa: 9-Box Yetenek Matrisi
# -------------------------------------------------
elif menu == "📊 9-Box Yetenek Matrisi":
    st.header("📊 9-Box Yetenek & Risk Matrisi")
    st.markdown("*Çalışanların performans-potansiyel oranlarını ayrılma riskleriyle çaprazlayarak İK stratejinizi belirleyin.*")
    
    df_db = load_employees_from_db(engine)
    processed_all = preprocess_input(df_db, feature_names)
    df_db['Risk_Skoru'] = model.predict_proba(processed_all)[:, 1]
    
    # Performans Puanı (1-4 -> 3 ve 4 ana veri, 1 ve 2 çok az/yok genelde IBM dataset'te, biz simüleli kullanacağız)
    # Performans = x-ekseni (Low: 1-2, Mid: 3, High: 4)
    df_db['Performans_Seviyesi'] = df_db['PerformanceRating'].map({1: 'Düşük', 2: 'Düşük', 3: 'Beklenen', 4: 'Üstün'})
    
    # Risk = y-ekseni (Low: 0-0.3, Mid: 0.3-0.6, High: 0.6-1.0)
    # 9-Box için ters çevirmemiz lazım çünkü yüksek risk "kötü", matriste genelde Y ekseni Potansiyeli gösterir.
    # Biz Y eksenine İş Memnuniyeti + Eğitim + Terfi gibi potansiyel/bağlılık karma metriği koyup, Renkleri Risk yapabiliriz.
    
    # Potansiyel Skoru Simülasyonu (0-100 arası karma bir skor)
    df_db['Potansiyel_Skoru'] = (df_db['Education'] / 5 * 30 + 
                                 df_db['JobInvolvement'] / 4 * 40 + 
                                 (10 - df_db['YearsSinceLastPromotion'].clip(upper=10)) / 10 * 30)

    df_db['Potansiyel_Kat'] = pd.cut(df_db['Potansiyel_Skoru'], bins=[0, 45, 75, 100], labels=['Düşük', 'Orta', 'Yüksek'])
    df_db['Risk_Kat'] = pd.cut(df_db['Risk_Skoru'], bins=[0, 0.3, 0.6, 1.0], labels=['Güvenli', 'Riskli', 'Kritik Değer'])

    col_mat, col_info = st.columns([3, 2])
    
    with col_mat:
        fig_scatter = px.scatter(df_db, x='PerformanceRating', y='Potansiyel_Skoru', color='Risk_Skoru',
                                 color_continuous_scale='RdYlGn_r',
                                 size='MonthlyIncome', hover_name='JobRole', 
                                 hover_data=['EmployeeNumber', 'Department', 'JobSatisfaction'],
                                 title="Performans vs Potansiyel (Renk: Ayrılma Riski, Boyut: Maaş)")
        # 9-Box ızgara çizgileri
        fig_scatter.add_vline(x=3.5, line_width=1, line_dash='dash', line_color='gray')
        fig_scatter.add_hline(y=45, line_width=1, line_dash='dash', line_color='gray')
        fig_scatter.add_hline(y=75, line_width=1, line_dash='dash', line_color='gray')
        fig_scatter.update_layout(xaxis_title="Performans Notu (3=Beklenen, 4=Üstün)", yaxis_title="Kariyer Potansiyel Skoru", height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_info:
        st.markdown("### 🏆 Yetenek Segmentleri")
        flight_risks = df_db[(df_db['PerformanceRating'] == 4) & (df_db['Risk_Skoru'] > 0.5)]
        core_stars = df_db[(df_db['PerformanceRating'] == 4) & (df_db['Risk_Skoru'] <= 0.3)]
        
        st.error(f"🚨 **Kaçış Riski Yüksek Yetenekler (Flight Risks):** {len(flight_risks)} kişi  \n"
                 f"*Yüksek performanslı ama kaybedilme ihtimali yüksek çalışanlar.*")
        st.success(f"🌟 **Çekirdek Yıldızlar (Core Stars):** {len(core_stars)} kişi  \n"
                   f"*Yüksek performanslı, şirkete bağlı ve potansiyeli yüksek kilit kadro.*")
        
        # Klasik 9-Box Tablosu (Heatmap View)
        st.markdown("#### Geleneksel 9-Box Görünümü (Risk Odaklı)")
        box_matrix = df_db.pivot_table(index='Risk_Kat', columns='Performans_Seviyesi', values='EmployeeNumber', aggfunc='count', observed=False).fillna(0)
        fig_heat = px.imshow(box_matrix, text_auto=True, color_continuous_scale='Reds', aspect='auto')
        fig_heat.update_layout(height=250, xaxis_title="Performans", yaxis_title="Risk Skoru")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.subheader("⚠️ Yüksek Riskli 'Üstün Performanslı' Çalışanlar (Acil Aksiyon)")
    display_cols = ['EmployeeNumber', 'Department', 'JobRole', 'MonthlyIncome', 'JobSatisfaction', 'Risk_Skoru']
    critical_stars = flight_risks[display_cols].sort_values('Risk_Skoru', ascending=False)
    st.dataframe(critical_stars.style.format({'Risk_Skoru': '{:.3f}', 'MonthlyIncome': '${:,.0f}'})
                 .background_gradient(subset=['Risk_Skoru'], cmap='Reds'), use_container_width=True)

# -------------------------------------------------
# 8. Sayfa: Tahmin & What-If
# -------------------------------------------------
elif menu == "🔮 Tahmin & What-If":
    st.header("🔮 İnteraktif Kariyer & Risk Simülasyonu (What-If)")
    st.markdown("*Bir çalışanın çalışma şartları değiştirildiğinde ayrılma riskinin nasıl tepki vereceğini gerçek zamanlı simüle edin.*")
    
    df_db_list = load_employees_from_db(engine)
    emp_ids = df_db_list['EmployeeNumber'].tolist()
    
    col_sel, _ = st.columns([1, 2])
    selected_emp_id = col_sel.selectbox("Profilini Simüle Edeceğiniz Çalışanı Seçin", emp_ids)
    full_emp_data = load_single_employee(engine, selected_emp_id)
    
    current_prob = model.predict_proba(preprocess_input(full_emp_data, feature_names))[0][1]
    
    st.markdown("---")
    sim_col1, sim_col2 = st.columns([1, 1])
    
    with sim_col1:
        st.subheader("🛠️ Parametreleri Değiştir")
        new_income = st.slider("💰 Aylık Maaş ($)", 1000, 25000, int(full_emp_data['MonthlyIncome'].iloc[0]), step=500)
        new_role = st.selectbox("👔 İş Rolü (Terfi/Rotasyon)", df_db_list['JobRole'].unique().tolist(), index=df_db_list['JobRole'].unique().tolist().index(full_emp_data['JobRole'].iloc[0]))
        
        c1, c2 = st.columns(2)
        new_ot = c1.radio("⏰ Fazla Mesai", ["Yes", "No"], index=0 if full_emp_data['OverTime'].iloc[0] == "Yes" else 1)
        new_sat = c2.select_slider("😊 İş Memnuniyeti", options=[1,2,3,4], value=int(full_emp_data['JobSatisfaction'].iloc[0]))
        
        new_dist = st.slider("🚗 Eve Uzaklık (km)", 1, 30, int(full_emp_data['DistanceFromHome'].iloc[0]))
        
        # Orijinal veriyi kopyala ve simüle değerleri ata
        what_if_data = full_emp_data.copy()
        what_if_data['MonthlyIncome'] = new_income
        what_if_data['JobRole'] = new_role
        what_if_data['OverTime'] = new_ot
        what_if_data['JobSatisfaction'] = new_sat
        what_if_data['DistanceFromHome'] = new_dist
        
        sim_prob = model.predict_proba(preprocess_input(what_if_data, feature_names))[0][1]
        
    with sim_col2:
        st.subheader("📊 Simülasyon Sonucu")
        
        # Gauge Chart yapımı (Mevcut vs Simüle)
        fig_gauge = go.Figure()
        
        fig_gauge.add_trace(go.Indicator(
            mode = "number+delta+gauge",
            value = sim_prob * 100,
            number = {'suffix': "%"},
            delta = {'reference': current_prob * 100, 'position': "top", 'valueformat': ".1f", 'suffix': "%"},
            title = {'text': "Yeni Ayrılma Riski", 'font': {'size': 20}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "rgba(0,0,0,0)"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(46, 204, 113, 0.6)"},
                    {'range': [30, 60], 'color': "rgba(243, 156, 18, 0.6)"},
                    {'range': [60, 100], 'color': "rgba(231, 76, 60, 0.6)"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': sim_prob * 100}
            }
        ))
        # Orijinal riski göstermek için işaret bırak
        fig_gauge.add_trace(go.Indicator(
            mode = "gauge", value = current_prob * 100, domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {'shape': "bullet", 'axis': {'range': [0,100], 'visible': False}, 'bar': {'color': "black", 'thickness': 0.1}}
        ))
        
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        curr_cost = calculate_costs(current_prob, full_emp_data['MonthlyIncome'].iloc[0])
        sim_cost = calculate_costs(sim_prob, new_income)
        
        rc1, rc2 = st.columns(2)
        rc1.metric("Mevcut Finansal Risk", f"${curr_cost:,.0f}")
        rc2.metric("Simüle Finansal Risk", f"${sim_cost:,.0f}", delta=f"${sim_cost - curr_cost:,.0f}", delta_color="inverse")
        
        if st.button("� Bu Senaryoyu Rapor Olarak İndir (PDF)"):
            pdf_path = generate_pdf_report(full_emp_data, sim_prob, sim_cost, [])
            with open(pdf_path, "rb") as f:
                st.download_button("� PDF İndir", f, f"WhatIf_Rapor_{selected_emp_id}.pdf")

# -------------------------------------------------
# 9. Sayfa: Müdahale & ROI
# -------------------------------------------------
elif menu == "💰 Müdahale & ROI Analizi":
    st.header("💰 Müdahale Aksiyonları ve ROI (Yatırım Getirisi) Analizi")
    st.markdown("*Belirli bir çalışana yapılacak müdahale paketlerini (maaş zammı, mentorluk, mesai iptali) bütçelendirip finansal getirilerini (ROI) test edin.*")
    
    df_db_list = load_employees_from_db(engine)
    selected_emp_id = st.selectbox("Çalışan Seç", df_db_list['EmployeeNumber'].tolist())
    full_emp_data = load_single_employee(engine, selected_emp_id)
    
    prob_curr = model.predict_proba(preprocess_input(full_emp_data, feature_names))[0][1]
    curr_salary = full_emp_data['MonthlyIncome'].iloc[0]
    loss_curr = calculate_costs(prob_curr, curr_salary)
    
    st.markdown("---")
    st.subheader("📦 Müdahale Paketini Oluştur")
    
    col_pax1, col_pax2, col_pax3 = st.columns(3)
    inc = col_pax1.number_input("💸 Yıllık Ekstra Zam ($)", min_value=0, max_value=50000, step=1000, value=0)
    ot_off = col_pax2.checkbox("⏰ Fazla Mesaiyi İptal Et (Yan Hak / Esnek)", value=False)
    sat_boost = col_pax3.checkbox("🎓 Özel Eğitim & Mentorluk Paketi", value=False)
    
    # Yeni durumu simüle et
    int_data = full_emp_data.copy()
    
    int_data['MonthlyIncome'] = curr_salary + (inc / 12)  # Aylığa vur
    if ot_off: int_data['OverTime'] = 'No'
    if sat_boost: 
        int_data['JobSatisfaction'] = min(4, int_data['JobSatisfaction'].iloc[0] + 1)
        int_data['EnvironmentSatisfaction'] = min(4, int_data['EnvironmentSatisfaction'].iloc[0] + 1)
        
    prob_int = model.predict_proba(preprocess_input(int_data, feature_names))[0][1]
    loss_int = calculate_costs(prob_int, int_data['MonthlyIncome'].iloc[0])

    # Maliyet Kalemleri
    cost_salary = inc
    cost_ot_adj = 2500 if ot_off else 0  # Taşere etme veya yan haklar maliyeti
    cost_edu = 1500 if sat_boost else 0
    total_investment = cost_salary + cost_ot_adj + cost_edu
    
    saving = loss_curr - loss_int
    roi = calculate_roi(saving, total_investment) if total_investment > 0 else 0

    col_res1, col_res2 = st.columns([2, 3])
    with col_res1:
        st.markdown("#### 🎯 Etki & Getiri Özeti")
        st.metric("Risk Düşüşü", f"%{prob_curr*100:.1f} ➡️ %{prob_int*100:.1f}", f"{-((prob_curr-prob_int)*100):.1f}%")
        st.metric("Toplam Yatırım (Maliyet)", f"${total_investment:,.0f}")
        st.metric("Önlenen Zarar (Tasarruf)", f"${saving:,.0f}")
        
        if total_investment > 0:
            if roi > 0:
                st.success(f"**Yatırım Getirisi (ROI): %{roi:.1f}**  \nHarcanan her 1 dolar size {(saving/total_investment):.1f} dolar tasarruf olarak dönüyor.")
            else:
                st.error(f"**Yatırım Getirisi (ROI): %{roi:.1f}**  \nBu müdahale paketi şirket için karlı değil. Daha düşük maliyetli seçenekler düşünün.")
        else:
            st.info("Henüz bir müdahale yatırım bedeli girilmedi.")

    with col_res2:
        st.markdown("#### 📉 Yatırım / Tasarruf Şelalesi (Waterfall)")
        fig_waterfall = go.Figure(go.Waterfall(
            name = "ROI Analizi", orientation = "v",
            measure = ["absolute", "relative", "relative", "relative", "total"],
            x = ["Mevcut Potansiyel Kayıp", "Maaş Artış Maliyeti", "Mesai/İzin Maliyeti", "Eğitim Maliyeti", "Yeni Potansiyel Kayıp"],
            textposition = "outside",
            text = [f"${loss_curr:,.0f}", f"${cost_salary:,.0f}", f"${cost_ot_adj:,.0f}", f"${cost_edu:,.0f}", f"${loss_int:,.0f}"],
            y = [loss_curr, cost_salary, cost_ot_adj, cost_edu, -loss_curr + (cost_salary + cost_ot_adj + cost_edu) + loss_int],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        # Renk ayarları (maliyetleri kırmızı yapmak)
        fig_waterfall.update_traces(decreasing_marker_color="red", increasing_marker_color="orange", totals_marker_color="green")
        fig_waterfall.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_waterfall, use_container_width=True)

# -------------------------------------------------
# 10. Sayfa: Kıyaslama
# -------------------------------------------------
elif menu == "👯 Çalışan Kıyaslama":
    st.header("👯 Çalışan Kıyaslaması")
    st.markdown("*İki çalışanı çoklu boyutlarda karşılaştırarak derinlikli analiz yapın*")
    
    df_db = load_employees_from_db(engine)
    processed_all = preprocess_input(df_db, feature_names)
    df_db['Risk'] = model.predict_proba(processed_all)[:, 1]
    
    # ── Çalışan Seçimi ──
    sel1, sel2 = st.columns(2)
    emp1 = sel1.selectbox("🔵 1. Çalışan", df_db['EmployeeNumber'].tolist(), index=0)
    emp2 = sel2.selectbox("🟠 2. Çalışan", df_db['EmployeeNumber'].tolist(), index=min(1, len(df_db)-1))
    
    d1 = df_db[df_db['EmployeeNumber'] == emp1].iloc[0]
    d2 = df_db[df_db['EmployeeNumber'] == emp2].iloc[0]
    
    # ── Profil Kartları ──
    st.subheader("👤 Çalışan Profilleri")
    pc1, pc2 = st.columns(2)
    
    with pc1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; color: white;">
            <h3 style="color: white; margin:0;">🔵 Çalışan #{emp1}</h3>
            <p><b>Rol:</b> {d1['JobRole']}<br>
            <b>Departman:</b> {d1['Department']}<br>
            <b>Yaş:</b> {int(d1['Age'])} | <b>Kıdem:</b> {int(d1['YearsAtCompany'])} yıl<br>
            <b>Maaş:</b> ${d1['MonthlyIncome']:,.0f}/ay<br>
            <b>Fazla Mesai:</b> {'✅ Evet' if d1['OverTime'] == 'Yes' else '❌ Hayır'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with pc2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 12px; color: white;">
            <h3 style="color: white; margin:0;">🟠 Çalışan #{emp2}</h3>
            <p><b>Rol:</b> {d2['JobRole']}<br>
            <b>Departman:</b> {d2['Department']}<br>
            <b>Yaş:</b> {int(d2['Age'])} | <b>Kıdem:</b> {int(d2['YearsAtCompany'])} yıl<br>
            <b>Maaş:</b> ${d2['MonthlyIncome']:,.0f}/ay<br>
            <b>Fazla Mesai:</b> {'✅ Evet' if d2['OverTime'] == 'Yes' else '❌ Hayır'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ── Risk Karşılaştırması ──
    st.subheader("🎯 Risk Karşılaştırması")
    rc1, rc2, rc3 = st.columns(3)
    risk1 = d1['Risk']
    risk2 = d2['Risk']
    cost1 = calculate_costs(risk1, d1['MonthlyIncome'])
    cost2 = calculate_costs(risk2, d2['MonthlyIncome'])
    
    rc1.metric(f"Çalışan #{emp1} Riski", f"%{risk1*100:.1f}", 
               delta=f"{'Yüksek' if risk1 > 0.5 else 'Düşük'} Risk", 
               delta_color="inverse" if risk1 > 0.5 else "normal")
    rc2.metric(f"Çalışan #{emp2} Riski", f"%{risk2*100:.1f}", 
               delta=f"{'Yüksek' if risk2 > 0.5 else 'Düşük'} Risk",
               delta_color="inverse" if risk2 > 0.5 else "normal")
    risk_diff = abs(risk1 - risk2) * 100
    more_risky = emp1 if risk1 > risk2 else emp2
    rc3.metric("Risk Farkı", f"%{risk_diff:.1f}", 
               delta=f"#{more_risky} daha riskli", delta_color="inverse")
    
    st.markdown("---")
    
    # ── 8 Boyutlu Radar Chart ──
    st.subheader("📊 Çok Boyutlu Karşılaştırma")
    
    radar_col, table_col = st.columns([3, 2])
    
    with radar_col:
        # Normalize edilen metrikler (0-5 arası)
        max_income = df_db['MonthlyIncome'].max()
        max_years = df_db['TotalWorkingYears'].max()
        max_company = df_db['YearsAtCompany'].max()
        max_distance = df_db['DistanceFromHome'].max()
        max_daily = df_db['DailyRate'].max()
        
        categories = ['Maaş', 'İş Memnuniyeti', 'Toplam Deneyim', 'Şirket Kıdemi', 
                       'Çevre Memnuniyeti', 'İlişki Memnuniyeti', 'Work-Life Balance', 'Performans']
        
        vals1 = [
            d1['MonthlyIncome'] / max_income * 5,
            d1['JobSatisfaction'] / 4 * 5,
            d1['TotalWorkingYears'] / max_years * 5 if max_years > 0 else 0,
            d1['YearsAtCompany'] / max_company * 5 if max_company > 0 else 0,
            d1['EnvironmentSatisfaction'] / 4 * 5,
            d1['RelationshipSatisfaction'] / 4 * 5,
            d1['WorkLifeBalance'] / 4 * 5,
            d1['PerformanceRating'] / 4 * 5
        ]
        vals2 = [
            d2['MonthlyIncome'] / max_income * 5,
            d2['JobSatisfaction'] / 4 * 5,
            d2['TotalWorkingYears'] / max_years * 5 if max_years > 0 else 0,
            d2['YearsAtCompany'] / max_company * 5 if max_company > 0 else 0,
            d2['EnvironmentSatisfaction'] / 4 * 5,
            d2['RelationshipSatisfaction'] / 4 * 5,
            d2['WorkLifeBalance'] / 4 * 5,
            d2['PerformanceRating'] / 4 * 5
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=vals1 + [vals1[0]], theta=categories + [categories[0]],
            fill='toself', name=f'Çalışan #{emp1}',
            line=dict(color='#667eea', width=2), fillcolor='rgba(102, 126, 234, 0.25)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=vals2 + [vals2[0]], theta=categories + [categories[0]],
            fill='toself', name=f'Çalışan #{emp2}',
            line=dict(color='#f5576c', width=2), fillcolor='rgba(245, 87, 108, 0.25)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            title="8 Boyutlu Yetkinlik Karşılaştırması",
            showlegend=True, height=500
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with table_col:
        st.markdown("##### 📋 Detaylı Metrik Tablosu")
        compare_data = {
            'Metrik': ['Aylık Maaş ($)', 'Yaş', 'Toplam Deneyim (Yıl)', 'Şirket Kıdemi (Yıl)',
                       'İş Memnuniyeti (1-4)', 'Çevre Memnuniyeti (1-4)', 'İlişki Memnuniyeti (1-4)',
                       'Work-Life Balance (1-4)', 'Performans (3-4)', 'Eğitim Düzeyi (1-5)',
                       'Eve Uzaklık (km)', 'Günlük Ücret ($)', 'Fazla Mesai',
                       'Ayrılma Riski (%)', 'Tahmini Kayıp ($)'],
            f'#{emp1}': [
                f"${d1['MonthlyIncome']:,.0f}", int(d1['Age']), int(d1['TotalWorkingYears']),
                int(d1['YearsAtCompany']), int(d1['JobSatisfaction']), int(d1['EnvironmentSatisfaction']),
                int(d1['RelationshipSatisfaction']), int(d1['WorkLifeBalance']),
                int(d1['PerformanceRating']), int(d1['Education']),
                int(d1['DistanceFromHome']), int(d1['DailyRate']),
                '✅ Evet' if d1['OverTime'] == 'Yes' else '❌ Hayır',
                f"%{risk1*100:.1f}", f"${cost1:,.0f}"
            ],
            f'#{emp2}': [
                f"${d2['MonthlyIncome']:,.0f}", int(d2['Age']), int(d2['TotalWorkingYears']),
                int(d2['YearsAtCompany']), int(d2['JobSatisfaction']), int(d2['EnvironmentSatisfaction']),
                int(d2['RelationshipSatisfaction']), int(d2['WorkLifeBalance']),
                int(d2['PerformanceRating']), int(d2['Education']),
                int(d2['DistanceFromHome']), int(d2['DailyRate']),
                '✅ Evet' if d2['OverTime'] == 'Yes' else '❌ Hayır',
                f"%{risk2*100:.1f}", f"${cost2:,.0f}"
            ]
        }
        st.dataframe(pd.DataFrame(compare_data).set_index('Metrik'), use_container_width=True, height=550)
    
    st.markdown("---")
    
    # ── Bar Chart Karşılaştırma ──
    st.subheader("📈 Metrik Bazlı Karşılaştırma")
    bar_metrics = ['MonthlyIncome', 'Age', 'TotalWorkingYears', 'YearsAtCompany', 
                   'DailyRate', 'DistanceFromHome', 'JobSatisfaction', 'EnvironmentSatisfaction']
    bar_labels = ['Maaş ($)', 'Yaş', 'Toplam Deneyim', 'Şirket Kıdemi',
                  'Günlük Ücret', 'Eve Uzaklık', 'İş Memnuniyeti', 'Çevre Memnuniyeti']
    
    bar_data = pd.DataFrame({
        'Metrik': bar_labels * 2,
        'Değer': [d1[m] for m in bar_metrics] + [d2[m] for m in bar_metrics],
        'Çalışan': [f'#{emp1}'] * len(bar_metrics) + [f'#{emp2}'] * len(bar_metrics)
    })
    
    fig_bar = px.bar(bar_data, x='Metrik', y='Değer', color='Çalışan', barmode='group',
                     color_discrete_map={f'#{emp1}': '#667eea', f'#{emp2}': '#f5576c'},
                     title="Yan Yana Metrik Karşılaştırması")
    fig_bar.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # ── Departman Ortalamasıyla Kıyaslama ──
    st.subheader("🏢 Departman Ortalamasına Göre Konumlandırma")
    
    dep_col1, dep_col2 = st.columns(2)
    
    for dep_col, d, emp_id, color in [(dep_col1, d1, emp1, '#667eea'), (dep_col2, d2, emp2, '#f5576c')]:
        with dep_col:
            dept = d['Department']
            dept_data = df_db[df_db['Department'] == dept]
            dept_avg_income = dept_data['MonthlyIncome'].mean()
            dept_avg_risk = dept_data['Risk'].mean()
            dept_avg_satisfaction = dept_data['JobSatisfaction'].mean()
            
            position_metrics = {
                'Maaş': (d['MonthlyIncome'], dept_avg_income),
                'Risk': (d['Risk'], dept_avg_risk),
                'Memnuniyet': (d['JobSatisfaction'], dept_avg_satisfaction)
            }
            
            st.markdown(f"**Çalışan #{emp_id}** — {dept} Departmanı")
            for metric, (val, avg) in position_metrics.items():
                if metric == 'Maaş':
                    diff_pct = ((val - avg) / avg * 100) if avg > 0 else 0
                    icon = "🟢" if diff_pct >= 0 else "🔴"
                    st.markdown(f"{icon} **{metric}:** ${val:,.0f} (Dept. ort: ${avg:,.0f}, fark: {diff_pct:+.1f}%)")
                elif metric == 'Risk':
                    icon = "🟢" if val <= avg else "🔴"
                    st.markdown(f"{icon} **{metric}:** %{val*100:.1f} (Dept. ort: %{avg*100:.1f})")
                else:
                    icon = "🟢" if val >= avg else "🔴"
                    st.markdown(f"{icon} **{metric}:** {val:.0f}/4 (Dept. ort: {avg:.1f})")
    
    st.markdown("---")
    
    # ── Güçlü ve Zayıf Yönler ──
    st.subheader("💡 Güçlü ve Zayıf Yönler Analizi")
    
    sw1, sw2 = st.columns(2)
    
    for sw_col, d, emp_id in [(sw1, d1, emp1), (sw2, d2, emp2)]:
        with sw_col:
            st.markdown(f"**Çalışan #{emp_id}**")
            strengths = []
            weaknesses = []
            
            # Maaş analizi
            if d['MonthlyIncome'] >= df_db['MonthlyIncome'].median():
                strengths.append("💰 Piyasa ortalamasının üzerinde maaş")
            else:
                weaknesses.append("💰 Piyasa ortalamasının altında maaş")
            
            # Memnuniyet
            if d['JobSatisfaction'] >= 3:
                strengths.append("😊 Yüksek iş memnuniyeti")
            else:
                weaknesses.append("😟 Düşük iş memnuniyeti")
            
            # Work-Life Balance
            if d['WorkLifeBalance'] >= 3:
                strengths.append("⚖️ İyi iş-yaşam dengesi")
            else:
                weaknesses.append("⚖️ Zayıf iş-yaşam dengesi")
            
            # Fazla Mesai
            if d['OverTime'] == 'No':
                strengths.append("🕐 Normal mesai saatleri")
            else:
                weaknesses.append("🕐 Fazla mesai yükü")
            
            # Çevre Memnuniyeti
            if d['EnvironmentSatisfaction'] >= 3:
                strengths.append("🏢 İyi çalışma ortamı memnuniyeti")
            else:
                weaknesses.append("🏢 Düşük çalışma ortamı memnuniyeti")
            
            # İlişki Memnuniyeti
            if d['RelationshipSatisfaction'] >= 3:
                strengths.append("🤝 İyi iş ilişkileri")
            else:
                weaknesses.append("🤝 Zayıf iş ilişkileri")
            
            # Kıdem
            if d['YearsAtCompany'] >= 5:
                strengths.append(f"📅 Deneyimli kadro ({int(d['YearsAtCompany'])} yıl)")
            elif d['YearsAtCompany'] <= 1:
                weaknesses.append("📅 Düşük kıdem (adaptasyon riski)")
            
            if strengths:
                st.success("**Güçlü Yönler:**\n" + "\n".join(f"- {s}" for s in strengths))
            if weaknesses:
                st.error("**Zayıf Yönler:**\n" + "\n".join(f"- {w}" for w in weaknesses))
            
            # Öneri
            risk_val = d['Risk']
            if risk_val > 0.5:
                st.warning(f"⚠️ **Aksiyon Önerisi:** Risk %{risk_val*100:.0f} — Acil müdahale gerekli. "
                          f"{'Fazla mesai kaldırılmalı. ' if d['OverTime'] == 'Yes' else ''}"
                          f"{'Maaş iyileştirmesi yapılmalı. ' if d['MonthlyIncome'] < df_db['MonthlyIncome'].median() else ''}"
                          f"{'Kariyer gelişim görüşmesi planlanmalı.' if d['JobSatisfaction'] <= 2 else ''}")
            else:
                st.info(f"✅ Risk %{risk_val*100:.0f} — Çalışan stabil görünüyor. Mevcut koşullar korunmalı.")

# -------------------------------------------------
# 11. Sayfa: Strateji Uzmanı
# -------------------------------------------------
elif menu == "🤖 Strateji Uzmanı":
    st.header("🤖 Yerel Strateji Uzmanı")
    st.markdown("*Model verileri ve istatistiksel analizlerle otomatik üretilen stratejik içgörüler*")
    
    df_db = load_employees_from_db(engine)
    processed = preprocess_input(df_db, feature_names)
    df_db['Risk'] = model.predict_proba(processed)[:, 1]
    
    # ── Genel Şirket Risk Özeti ──
    avg_risk = df_db['Risk'].mean()
    high_risk_count = len(df_db[df_db['Risk'] > 0.5])
    critical_count = len(df_db[df_db['Risk'] > 0.7])
    total_potential_loss = df_db.apply(lambda x: calculate_costs(x['Risk'], x['MonthlyIncome']), axis=1).sum()
    
    st.subheader("📊 Şirket Geneli Risk Özeti")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Şirket Risk Ortalaması", f"%{avg_risk*100:.1f}")
    m2.metric("Yüksek Riskli (>%50)", f"{high_risk_count} kişi")
    m3.metric("Kritik Riskli (>%70)", f"{critical_count} kişi", delta=f"%{critical_count/len(df_db)*100:.1f} oran", delta_color="inverse")
    m4.metric("Toplam Potansiyel Kayıp", f"${total_potential_loss:,.0f}")
    
    st.markdown("---")
    
    # ── Risk Dağılım Grafiği ──
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig_hist = px.histogram(df_db, x='Risk', nbins=30, title="Risk Skoru Dağılımı",
                                color_discrete_sequence=['#e74c3c'], opacity=0.8)
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="orange", annotation_text="Yüksek Risk Eşiği")
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="Kritik Eşik")
        fig_hist.update_layout(xaxis_title="Risk Skoru", yaxis_title="Çalışan Sayısı")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col_chart2:
        risk_cats = pd.cut(df_db['Risk'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                           labels=['Çok Düşük', 'Düşük', 'Orta', 'Yüksek', 'Kritik'])
        risk_dist = risk_cats.value_counts().sort_index()
        fig_pie = px.pie(values=risk_dist.values, names=risk_dist.index, title="Risk Kategorisi Dağılımı",
                         color_discrete_sequence=['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c'])
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # ── 1. Maaş Analizi ──
    with st.expander("💰 Maaş ve Ücret Analizi", expanded=True):
        low_inc = df_db[(df_db['MonthlyIncome'] < df_db['MonthlyIncome'].median()) & (df_db['Risk'] > 0.4)]
        high_inc = df_db[(df_db['MonthlyIncome'] >= df_db['MonthlyIncome'].median()) & (df_db['Risk'] > 0.4)]
        
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Düşük Maaşlı Riskli Çalışan", f"{len(low_inc)} kişi")
        sc2.metric("Yüksek Maaşlı Riskli Çalışan", f"{len(high_inc)} kişi")
        sc3.metric("Medyan Maaş", f"${df_db['MonthlyIncome'].median():,.0f}")
        
        # Maaş-Risk korelasyonu
        fig_scatter = px.scatter(df_db, x='MonthlyIncome', y='Risk', color='Department',
                                 title="Maaş vs Risk Skoru", opacity=0.6,
                                 labels={'MonthlyIncome': 'Aylık Gelir ($)', 'Risk': 'Ayrılma Riski'})
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        if len(low_inc) > 0:
            estimated_raise_cost = low_inc['MonthlyIncome'].sum() * 0.10 * 12
            prevented_loss = low_inc.apply(lambda x: calculate_costs(x['Risk'], x['MonthlyIncome']), axis=1).sum()
            st.warning(f"⚠️ **{len(low_inc)} düşük maaşlı riskli çalışan** tespit edildi.")
            st.info(f"📋 **Öneri:** %10 maaş artışı → Yıllık maliyet: **${estimated_raise_cost:,.0f}** | Önlenen potansiyel kayıp: **${prevented_loss:,.0f}** | **ROI: %{((prevented_loss - estimated_raise_cost) / estimated_raise_cost * 100) if estimated_raise_cost > 0 else 0:.0f}**")
        else:
            st.success("Düşük maaşlı riskli çalışan bulunmamaktadır.")
    
    # ── 2. Fazla Mesai Analizi ──
    with st.expander("⏰ Fazla Mesai Analizi"):
        ot_yes = df_db[df_db['OverTime'] == 'Yes']
        ot_no = df_db[df_db['OverTime'] == 'No']
        
        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Fazla Mesai Yapanlar", f"{len(ot_yes)} kişi (%{len(ot_yes)/len(df_db)*100:.0f})")
        oc2.metric("FM Risk Ortalaması", f"%{ot_yes['Risk'].mean()*100:.1f}" if len(ot_yes) > 0 else "N/A")
        oc3.metric("Normal Risk Ortalaması", f"%{ot_no['Risk'].mean()*100:.1f}" if len(ot_no) > 0 else "N/A")
        
        # Fazla mesai karşılaştırma
        ot_compare = pd.DataFrame({
            'Grup': ['Fazla Mesai Var', 'Fazla Mesai Yok'],
            'Ortalama Risk': [ot_yes['Risk'].mean() if len(ot_yes) > 0 else 0, 
                              ot_no['Risk'].mean() if len(ot_no) > 0 else 0],
            'Kişi Sayısı': [len(ot_yes), len(ot_no)]
        })
        fig_ot = px.bar(ot_compare, x='Grup', y='Ortalama Risk', color='Grup',
                        color_discrete_map={'Fazla Mesai Var': '#e74c3c', 'Fazla Mesai Yok': '#2ecc71'},
                        title="Fazla Mesai ve Risk Karşılaştırması", text_auto='.3f')
        st.plotly_chart(fig_ot, use_container_width=True)
        
        if len(ot_yes) > 0 and ot_yes['Risk'].mean() > ot_no['Risk'].mean():
            risk_ratio = ot_yes['Risk'].mean() / ot_no['Risk'].mean() if ot_no['Risk'].mean() > 0 else 0
            risky_ot = ot_yes[ot_yes['Risk'] > 0.5]
            st.warning(f"⚠️ Fazla mesaiciler **{risk_ratio:.1f}x daha yüksek** ayrılma riskine sahip.")
            st.info(f"📋 **Öneri:** {len(risky_ot)} yüksek riskli fazla mesaiciye esnek çalışma modeli, TOIL (izin karşılığı) ve iş yükü dengeleme programı uygulanmalı.")
    
    # ── 3. Departman Bazlı Risk Analizi ──
    with st.expander("🏢 Departman Bazlı Risk Analizi"):
        dept_stats = df_db.groupby('Department').agg(
            Ortalama_Risk=('Risk', 'mean'),
            Yüksek_Risk_Sayısı=('Risk', lambda x: (x > 0.5).sum()),
            Çalışan_Sayısı=('Risk', 'count'),
            Ortalama_Maaş=('MonthlyIncome', 'mean')
        ).reset_index()
        dept_stats['Yüksek_Risk_Oranı'] = (dept_stats['Yüksek_Risk_Sayısı'] / dept_stats['Çalışan_Sayısı'] * 100).round(1)
        
        st.dataframe(dept_stats.style.background_gradient(subset=['Ortalama_Risk'], cmap='RdYlGn_r')
                     .format({'Ortalama_Risk': '{:.3f}', 'Ortalama_Maaş': '${:,.0f}', 'Yüksek_Risk_Oranı': '%{:.1f}'}),
                     use_container_width=True)
        
        riskiest_dept = dept_stats.loc[dept_stats['Ortalama_Risk'].idxmax()]
        st.warning(f"⚠️ En riskli departman: **{riskiest_dept['Department']}** (Ort. Risk: %{riskiest_dept['Ortalama_Risk']*100:.1f}, Yüksek riskli: {int(riskiest_dept['Yüksek_Risk_Sayısı'])} kişi)")
        st.info(f"📋 **Öneri:** {riskiest_dept['Department']} departmanında acil çalışan memnuniyeti anketi, birebir görüşme ve kariyer gelişim planı oluşturulmalı.")
    
    # ── 4. Yaş Grubu Analizi ──
    with st.expander("👥 Yaş Grubu Risk Analizi"):
        df_db['Yaş_Grubu'] = pd.cut(df_db['Age'], bins=[17, 25, 30, 35, 40, 50, 65],
                                     labels=['18-25', '26-30', '31-35', '36-40', '41-50', '51-65'])
        age_stats = df_db.groupby('Yaş_Grubu', observed=False).agg(
            Ortalama_Risk=('Risk', 'mean'),
            Kişi_Sayısı=('Risk', 'count'),
            Yüksek_Risk=('Risk', lambda x: (x > 0.5).sum())
        ).reset_index()
        
        fig_age = px.bar(age_stats, x='Yaş_Grubu', y='Ortalama_Risk', color='Ortalama_Risk',
                         color_continuous_scale='RdYlGn_r', title="Yaş Grubuna Göre Ortalama Risk",
                         text_auto='.3f')
        st.plotly_chart(fig_age, use_container_width=True)
        
        riskiest_age = age_stats.loc[age_stats['Ortalama_Risk'].idxmax()]
        st.warning(f"⚠️ En riskli yaş grubu: **{riskiest_age['Yaş_Grubu']}** (Ort. Risk: %{riskiest_age['Ortalama_Risk']*100:.1f}, {int(riskiest_age['Kişi_Sayısı'])} kişi)")
        st.info("📋 **Öneri:** Genç çalışanlara mentor ataması, kariyer yol haritası, eğitim bursları ve departman rotasyonu programı sunulmalı.")
    
    # ── 5. Kıdem ve Sadakat Analizi ──
    with st.expander("📅 Kıdem ve Sadakat Analizi"):
        df_db['Kıdem_Grubu'] = pd.cut(df_db['YearsAtCompany'], bins=[-1, 1, 3, 5, 10, 40],
                                       labels=['0-1 yıl', '2-3 yıl', '4-5 yıl', '6-10 yıl', '10+ yıl'])
        tenure_stats = df_db.groupby('Kıdem_Grubu', observed=False).agg(
            Ortalama_Risk=('Risk', 'mean'),
            Kişi_Sayısı=('Risk', 'count'),
            Ortalama_Maaş=('MonthlyIncome', 'mean')
        ).reset_index()
        
        fig_tenure = px.bar(tenure_stats, x='Kıdem_Grubu', y='Ortalama_Risk', color='Ortalama_Risk',
                            color_continuous_scale='RdYlGn_r', title="Kıdem Grubuna Göre Ortalama Risk",
                            text_auto='.3f')
        st.plotly_chart(fig_tenure, use_container_width=True)
        
        riskiest_tenure = tenure_stats.loc[tenure_stats['Ortalama_Risk'].idxmax()]
        st.warning(f"⚠️ En riskli kıdem grubu: **{riskiest_tenure['Kıdem_Grubu']}** (Ort. Risk: %{riskiest_tenure['Ortalama_Risk']*100:.1f})")
        st.info("📋 **Öneri:** İlk yıllarındaki çalışanlara oryantasyon güçlendirilmeli, buddy sistemi kurulmalı ve düzenli check-in toplantıları yapılmalı.")
    
    # ── 6. İş Memnuniyeti Analizi ──
    with st.expander("😊 İş Memnuniyeti ve Risk İlişkisi"):
        sat_stats = df_db.groupby('JobSatisfaction').agg(
            Ortalama_Risk=('Risk', 'mean'),
            Kişi_Sayısı=('Risk', 'count')
        ).reset_index()
        sat_stats['Memnuniyet_Seviyesi'] = sat_stats['JobSatisfaction'].map({1: '1 - Düşük', 2: '2 - Orta', 3: '3 - Yüksek', 4: '4 - Çok Yüksek'})
        
        fig_sat = px.bar(sat_stats, x='Memnuniyet_Seviyesi', y='Ortalama_Risk', color='Ortalama_Risk',
                         color_continuous_scale='RdYlGn_r', title="İş Memnuniyeti vs Ayrılma Riski",
                         text_auto='.3f')
        st.plotly_chart(fig_sat, use_container_width=True)
        
        low_sat_risky = df_db[(df_db['JobSatisfaction'] <= 2) & (df_db['Risk'] > 0.4)]
        st.warning(f"⚠️ {len(low_sat_risky)} çalışan hem düşük memnuniyetli hem yüksek riskli.")
        st.info("📋 **Öneri:** Düşük memnuniyetli çalışanlara birebir geri bildirim görüşmesi, çalışma koşulları iyileştirmesi ve kariyer danışmanlığı sunulmalı.")
    
    # ── Otomatik Strateji Raporu ──
    st.markdown("---")
    st.subheader("📝 Otomatik Strateji Raporu")
    
    strategies = []
    # Maaş stratejisi
    if len(low_inc) > 5:
        strategies.append(f"🔴 **Acil – Ücret İyileştirmesi:** {len(low_inc)} düşük maaşlı riskli çalışana %10 maaş artışı uygulanmalı. Tahmini yıllık maliyet: ${low_inc['MonthlyIncome'].sum()*0.10*12:,.0f}")
    # Fazla mesai stratejisi
    risky_ot_count = len(df_db[(df_db['OverTime'] == 'Yes') & (df_db['Risk'] > 0.5)])
    if risky_ot_count > 5:
        strategies.append(f"🟠 **Öncelikli – Fazla Mesai Yönetimi:** {risky_ot_count} fazla mesaici yüksek risk altında. Esnek çalışma ve izin karşılığı mesai uygulaması önerilir.")
    # Departman stratejisi
    strategies.append(f"🟡 **Departman Odaklı:** {riskiest_dept['Department']} departmanında özel tutundurma programı başlatılmalı.")
    # Yaş stratejisi
    young_risky = len(df_db[(df_db['Age'] < 30) & (df_db['Risk'] > 0.4)])
    if young_risky > 5:
        strategies.append(f"🔵 **Genç Yetenek Programı:** {young_risky} genç çalışan risk altında. Mentor programı ve kariyer gelişim planı oluşturulmalı.")
    # Genel
    strategies.append(f"🟢 **Genel:** Şirket geneli risk ortalaması %{avg_risk*100:.1f}. {'Risk seviyesi kabul edilebilir düzeyde.' if avg_risk < 0.25 else 'Risk seviyesi yüksek, acil aksiyon gerekli!' if avg_risk > 0.35 else 'Risk orta düzeyde, önleyici tedbirler sürdürülmeli.'}")
    
    for i, s in enumerate(strategies, 1):
        st.markdown(f"{i}. {s}")

# -------------------------------------------------
# 12. Sayfa: Model Şeffaflığı
# -------------------------------------------------
else:
    st.header("🔍 Model Karar Mekanizması (SHAP)")
    st.markdown("*Yapay zeka modelinin tahminlerini nasıl yaptığını anlayın*")
    
    # ── SHAP Nedir? Açıklaması ──
    with st.expander("❓ SHAP Nedir? (Tıklayın)", expanded=True):
        st.markdown("""
        ### SHAP (SHapley Additive exPlanations) — Basitçe Açıklama
        
        Yapay zeka modelleri genellikle **"kara kutu"** gibi çalışır — tahmin yapar ama nedenini açıklamaz.  
        **SHAP** bu kara kutuyu açarak modelin **her bir kararının arkasındaki nedenleri** gösterir.
        
        🎯 **Temel Soru:** *"Model bu çalışanın ayrılacağını neden düşünüyor?"*
        
        **Örnek ile açıklama:**
        - Model, Çalışan A'nın **%80** ihtimalle ayrılacağını tahmin etti
        - SHAP bize şunu söylüyor:
            - 📍 **Fazla mesai yapması** → Riski **+%25** artırıyor
            - 📍 **Düşük maaşı** → Riski **+%20** artırıyor  
            - 📍 **Yüksek kıdemi** → Riski **-%10** azaltıyor
        
        Bu sayede İK yöneticisi **hangi faktöre müdahale edeceğini** bilir!
        """)
    
    st.markdown("---")
    
    # ── SHAP Analizi ──
    try:
        df_v2 = pd.read_csv("data/hr_attrition_preprocessed_v2.csv").sample(100, random_state=42)
        X_viz = df_v2.drop(columns=['Attrition'])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_viz)
        
        # ── 1. SHAP Summary Plot ──
        st.subheader("📊 SHAP Özet Grafiği (Beeswarm Plot)")
        
        fig_shap, ax_shap = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_viz, show=False, max_display=15)
        st.pyplot(fig_shap)
        plt.close(fig_shap)
        
        # ── Grafik Okuma Kılavuzu ──
        st.markdown("---")
        st.subheader("📖 Bu Grafiği Nasıl Okumalı?")
        
        guide1, guide2 = st.columns(2)
        with guide1:
            st.markdown("""
            #### 🔴🔵 Renklerin Anlamı
            
            | Renk | Anlam |
            |------|-------|
            | 🔴 **Kırmızı** | O özelliğin değeri **yüksek** |
            | 🔵 **Mavi** | O özelliğin değeri **düşük** |
            
            **Örnek:** `MonthlyIncome` satırında:
            - 🔵 Mavi noktalar **sol** taraftaysa → Düşük maaş, riski **artırıyor**
            - 🔴 Kırmızı noktalar **sol** taraftaysa → Yüksek maaş, riski **azaltıyor**
            """)
        
        with guide2:
            st.markdown("""
            #### ◀️ ▶️ Konumların Anlamı
            
            | Konum | Anlam |
            |-------|-------|
            | **Sağ tarafa** giden noktalar | Riski **artıran** etki |
            | **Sol tarafa** giden noktalar | Riski **azaltan** etki |
            | **Sıfıra yakın** noktalar | Etkisi **az/yok** |
            
            **Örnek:** Bir nokta ne kadar sağdaysa,  
            o özellik o çalışanın ayrılma riskini  
            o kadar **artırıyor** demektir.
            """)
        
        st.info("💡 **İpucu:** Grafikteki her satır bir özelliktir. Üstten alta doğru **en etkili** özelllikten **en az etkili** özelliğe sıralanır.")
        
        st.markdown("---")
        
        # ── 2. Feature Importance Bar Chart ──
        st.subheader("📈 En Etkili 10 Özellik (Sıralama)")
        
        # SHAP değerlerinin mutlak ortalamasını hesapla
        if isinstance(shap_values, list):
            shap_importance = np.abs(shap_values[1]).mean(axis=0) if len(shap_values) > 1 else np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
        
        feat_imp = pd.DataFrame({
            'Özellik': X_viz.columns,
            'SHAP Etkisi': shap_importance
        }).sort_values('SHAP Etkisi', ascending=False).head(10)
        
        fig_bar = px.bar(feat_imp.iloc[::-1], x='SHAP Etkisi', y='Özellik', orientation='h',
                         color='SHAP Etkisi', color_continuous_scale='RdYlGn_r',
                         title="Modelin En Çok Dikkat Ettiği 10 Özellik")
        fig_bar.update_layout(height=450, yaxis_title="", xaxis_title="Ortalama SHAP Etkisi (Mutlak)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("☝️ *Bu grafik, modelin karar verirken en çok hangi özelliklere baktığını gösterir. Üstteki özellikler modelin kararını en çok etkileyen faktörlerdir.*")
        
        st.markdown("---")
        
        # ── 3. Özelliklerin Türkçe Açıklamaları ──
        st.subheader("📋 Özellik Açıklamaları ve İK Yorumları")
        
        feature_descriptions = {
            'MonthlyIncome': ('💰 Aylık Maaş', 'Çalışanın aylık brüt geliri. Düşük maaş genellikle yüksek ayrılma riski ile ilişkilidir.', '🔴 Maaşı medyanın altındaki çalışanlara ücret revizyonu uygulayın.'),
            'OverTime_Yes': ('⏰ Fazla Mesai', 'Çalışanın düzenli olarak fazla mesai yapıp yapmadığı. Fazla mesai, tükenmişlik ve ayrılma riskini artırır.', '🔴 Fazla mesaici sayısını izleyin, esnek çalışma modeli sunun.'),
            'Age': ('👤 Yaş', 'Çalışanın yaşı. Genç çalışanlar (25-35) daha yüksek mobiliteye sahiptir.', '🟡 Genç çalışanlara mentor ve kariyer yol haritası sunun.'),
            'TotalWorkingYears': ('📅 Toplam Çalışma Yılı', 'Çalışanın toplam iş deneyimi. Az deneyimli çalışanlarda risk yüksektir.', '🟡 Yeni mezunlara oryantasyon ve gelişim programı sunun.'),
            'DailyRate': ('💵 Günlük Ücret', 'Çalışanın günlük bazda hesaplanan ücreti.', '🟢 Düzenli piyasa araştırması ile rekabetçi ücret sağlayın.'),
            'YearsAtCompany': ('🏢 Şirketteki Yıl', 'Çalışanın mevcut şirketteki kıdemi. İlk 1-2 yıl kritik adaptasyon dönemidir.', '🔴 İlk yıllarındaki çalışanlara buddy sistemi ve check-in görüşmeleri yapın.'),
            'DistanceFromHome': ('🚗 Eve Uzaklık', 'Çalışanın evinden iş yerine olan mesafesi. Uzak mesafe iş tatminini düşürür.', '🟡 Uzak çalışanlara hibrit çalışma veya ulaşım desteği sunun.'),
            'JobSatisfaction': ('😊 İş Memnuniyeti', 'Çalışanın işinden ne kadar memnun olduğu (1-4 ölçeğinde). Düşük memnuniyet = yüksek risk.', '🔴 Memnuniyeti düşük çalışanlarla birebir görüşme yapın.'),
            'EnvironmentSatisfaction': ('🏠 Çevre Memnuniyeti', 'Çalışma ortamından duyulan memnuniyet. Fiziksel ortam, ekipman ve ofis koşulları.', '🟡 Çalışma koşullarını iyileştirin, ergonomik düzenleme yapın.'),
            'WorkLifeBalance': ('⚖️ İş-Yaşam Dengesi', 'Çalışanın iş ve özel hayat dengesini nasıl değerlendirdiği.', '🔴 Dengesizlik hisseden çalışanlara esnek saat ve uzaktan çalışma imkanı sunun.'),
            'YearsInCurrentRole': ('🎭 Mevcut Roldeki Yıl', 'Aynı pozisyonda geçirilen süre. Uzun süre aynı rolde kalmak motivasyonu düşürebilir.', '🟡 Kariyer rotasyonu ve yeni sorumluluklar sunun.'),
            'YearsSinceLastPromotion': ('📊 Son Terfiden Bu Yana', 'En son terfi alınan tarihten bu yana geçen yıl. Uzun süre terfi alamama riski artırır.', '🔴 Terfi süreçlerini şeffaf hale getirin, gelişim planı oluşturun.'),
            'NumCompaniesWorked': ('🔄 Çalışılan Şirket Sayısı', 'Daha önce çalışılan toplam şirket sayısı. Çok fazla şirket değiştirme eğilimi risk göstergesi olabilir.', '🟢 İşe alım sürecinde kariyer geçmişini değerlendirin.'),
            'Education': ('🎓 Eğitim Düzeyi', 'Çalışanın eğitim seviyesi (1: Lise altı → 5: Doktora).', '🟢 Eğitim düzeyine uygun pozisyonlama yapın.'),
            'RelationshipSatisfaction': ('🤝 İlişki Memnuniyeti', 'İş arkadaşları ve yöneticilerle ilişki kalitesi.', '🟡 Takım uyumu etkinlikleri ve iletişim eğitimleri düzenleyin.'),
            'Income_Per_Year': ('📈 Yıl Başına Kazanç', 'Aylık maaşın toplam deneyim yılına oranı. Tecrübeye göre adil ücret alıp almadığını gösterir.', '🔴 Tecrübesine göre düşük maaş alanları tespit edin.'),
            'Tenure_Ratio': ('📊 Sadakat Oranı', 'Şirketteki yılın toplam deneyime oranı. Düşük oran = çok şirket değiştirme eğilimi.', '🟡 Sadakati düşük profillerde bağlılık programları uygulayın.'),
            'PerformanceRating': ('⭐ Performans Notu', 'Çalışanın performans değerlendirmesi (3: Beklenen, 4: Üstün).', '🟢 Yüksek performanslı çalışanları ödüllendirin.'),
        }
        
        top_features = feat_imp['Özellik'].tolist()
        
        for i, feat_name in enumerate(top_features, 1):
            if feat_name in feature_descriptions:
                label, desc, action = feature_descriptions[feat_name]
                with st.expander(f"**{i}. {label}** — `{feat_name}`"):
                    st.markdown(f"**📝 Açıklama:** {desc}")
                    st.markdown(f"**💼 İK Aksiyon Önerisi:** {action}")
                    # Etkinin yönü
                    feat_shap = feat_imp[feat_imp['Özellik'] == feat_name]['SHAP Etkisi'].values[0]
                    st.metric("Ortalama SHAP Etkisi", f"{feat_shap:.4f}", 
                              delta="Güçlü Etki" if feat_shap > 0.05 else "Orta Etki" if feat_shap > 0.02 else "Hafif Etki")
            else:
                with st.expander(f"**{i}. {feat_name}**"):
                    st.markdown(f"**📝 Açıklama:** Bu özellik modelin kararını etkilemektedir.")
                    feat_shap = feat_imp[feat_imp['Özellik'] == feat_name]['SHAP Etkisi'].values[0]
                    st.metric("Ortalama SHAP Etkisi", f"{feat_shap:.4f}")
        
        st.markdown("---")
        
        # ── 4. Pratik Sonuçlar ──
        st.subheader("🎯 Pratik Sonuçlar — İK İçin Ne Anlama Geliyor?")
        
        top3 = top_features[:3]
        top3_labels = [feature_descriptions.get(f, (f, '', ''))[0] for f in top3]
        
        st.success(f"""
        ### Model bu 3 faktöre en çok dikkat ediyor:
        
        1. **{top3_labels[0]}** (`{top3[0]}`)
        2. **{top3_labels[1]}** (`{top3[1]}`)
        3. **{top3_labels[2]}** (`{top3[2]}`)
        
        Bu demek oluyor ki, bir çalışanın işten ayrılıp ayrılmayacağını tahmin ederken model en çok bu üç bilgiye bakıyor.
        **Bu alanlarda iyileştirme yapmak, çalışan kaybını en etkili şekilde önleyecektir.**
        """)
        
        st.markdown("""
        #### 📌 Aksiyon Planı Özeti
        
        | Öncelik | Aksiyon | Beklenen Etki |
        |---------|---------|---------------|
        | 🔴 **Acil** | Düşük maaşlı ve fazla mesaici çalışanlara müdahale | Yüksek etkili — risk skorlarını hızla düşürür |
        | 🟠 **Kısa Vade** | İş memnuniyeti ve çevre memnuniyeti anketleri | Orta etkili — sorun alanlarını tespit eder |
        | 🟡 **Orta Vade** | Kariyer gelişim planları ve mentor programları | Uzun vadeli bağlılık artışı |
        | 🟢 **Sürekli** | Düzenli check-in, şeffaf terfi politikası | Kurumsal kültür güçlendirme |
        """)
        
        st.markdown("---")
        
        # ── 5. Sözlük ──
        with st.expander("📚 Teknik Terimler Sözlüğü"):
            st.markdown("""
            | Terim | Açıklama |
            |-------|----------|
            | **SHAP** | Her tahmin için hangi faktörlerin ne kadar etkili olduğunu ölçen bir yöntem |
            | **SHAP Değeri** | Bir özelliğin tek bir tahmine olan katkısı (+ artırır, - azaltır) |
            | **Feature Importance** | Bir özelliğin genel olarak model için ne kadar önemli olduğu |
            | **Beeswarm Plot** | Her noktanın bir çalışanı temsil ettiği dağılım grafiği |
            | **XGBoost** | Modelin kullandığı güçlü makine öğrenmesi algoritması |
            | **Pozitif SHAP** | Ayrılma riskini **artıran** etki |
            | **Negatif SHAP** | Ayrılma riskini **azaltan** etki |
            """)
    
    except Exception as e:
        st.error(f"SHAP analizi sırasında bir hata oluştu: {e}")
        st.info("Lütfen 'hr_preprocessing_v2.py' ve 'hr_model_advanced_v2.py' scriptlerinin çalıştırıldığından emin olun.")

st.sidebar.markdown("---")
st.sidebar.markdown("Hasan Yiğit Doğanay | HR Analytics Project")
