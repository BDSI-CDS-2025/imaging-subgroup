# Cancer Data Science Project Portfolio

A comprehensive machine learning analysis of cancer biomarkers (ER, HER2, PR) developed as part of the BDSI Imaging Subgroup research project.

## 🌟 Live Portfolio

Visit the live portfolio at: `https://[your-username].github.io/[repository-name]`

## 📁 Project Structure

```
├── index.html                          # Main portfolio homepage
├── presentation.html                   # Reveal.js interactive presentation
├── final_symposium_poster.pdf         # Research poster
├── final_symposium_presentation.pdf   # Original presentation PDF
├── data/final/lucy/                   # Key research results
│   ├── figures/custom/                # Visualization outputs
│   ├── summary_auc.csv               # Model performance metrics
│   ├── all_feature_importance.csv    # Feature analysis results
│   └── *.py                          # Analysis scripts
└── README.md                         # This file
```

## 🚀 Setting up GitHub Pages

### Option 1: Using GitHub Web Interface

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Add portfolio website"
   git push origin main
   ```

2. **Enable GitHub Pages:**
   - Go to your repository on GitHub
   - Click on "Settings" tab
   - Scroll down to "Pages" section
   - Under "Source", select "Deploy from a branch"
   - Choose "main" branch and "/ (root)" folder
   - Click "Save"

3. **Access your site:**
   - Your site will be available at: `https://[username].github.io/[repository-name]`
   - It may take a few minutes to deploy

### Option 2: Using GitHub Actions (Recommended)

1. **Create GitHub Actions workflow:**
   ```bash
   mkdir -p .github/workflows
   ```

2. **Add the workflow file** (this will be created automatically below)

3. **Push and deploy:**
   ```bash
   git add .
   git commit -m "Add GitHub Pages workflow"
   git push origin main
   ```

## 🎯 Portfolio Features

### 📊 Interactive Sections

- **Overview**: Project methodology and objectives
- **Results**: Comprehensive analysis with visualizations for all three biomarkers
- **Poster**: Embedded PDF viewer for the research poster
- **Presentation**: Interactive Reveal.js presentation

### 🔬 Research Highlights

- **Machine Learning Models**: Comprehensive model zoo evaluation
- **Biomarker Analysis**: ER, HER2, and PR receptor prediction
- **Performance Metrics**: Detailed AUC analysis and feature importance
- **Visualizations**: Heatmaps, box plots, and bar charts

### 🎨 Design Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Modern UI**: Clean, professional design with smooth animations
- **Interactive Navigation**: Tabbed interface for easy exploration
- **Download Links**: Direct access to data files and PDFs

## 📈 Key Results

The project analyzes three critical cancer biomarkers:

1. **Estrogen Receptor (ER)**: Hormone receptor analysis
2. **HER2**: Growth factor receptor targeting
3. **Progesterone Receptor (PR)**: Treatment decision support

## 🛠️ Technologies Used

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Presentation**: Reveal.js framework
- **Visualizations**: Python-generated figures (matplotlib/seaborn)
- **Deployment**: GitHub Pages
- **Data Analysis**: Python (pandas, scikit-learn, etc.)

## 📱 Mobile Responsiveness

The portfolio is fully responsive and optimized for:
- Desktop computers
- Tablets
- Mobile phones
- Various screen sizes and orientations

## 🔧 Customization

To customize the portfolio:

1. **Update content** in `index.html`
2. **Modify presentation** in `presentation.html`
3. **Add new visualizations** to the results section
4. **Update styling** in the CSS sections

## 📞 Contact & Attribution

- **Project**: BDSI Imaging Subgroup
- **Focus**: Cancer Data Science Research
- **Framework**: Machine Learning Biomarker Analysis

## 🎓 Academic Context

This portfolio showcases research conducted as part of the BDSI (Biomedical Data Science Initiative) program, focusing on the application of machine learning techniques to cancer biomarker prediction and analysis.

---

**Built with ❤️ for scientific discovery and knowledge sharing.**
