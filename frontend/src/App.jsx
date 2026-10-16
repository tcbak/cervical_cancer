import { useState } from 'react'

import {

  Activity,

  Brain,

  ChartNoAxesCombined,

  CircleHelp,

  ClipboardList,

  FileImage,

  House,

  Settings,

  ShieldCheck,

  Upload,

  X,

} from 'lucide-react'

import './App.css'



function App() {

  const [activePage, setActivePage] = useState('Dashboard')

  const [imagePreview, setImagePreview] = useState(null)

  const [imageName, setImageName] = useState('')



  const [selectedFile, setSelectedFile] = useState(null)

  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const [analysisError, setAnalysisError] = useState('')

  const [results, setResults] = useState(null)







function handleImageUpload(event) {

  const file = event.target.files?.[0]

  if (!file) return



  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {

    setAnalysisError('Please select a PNG, JPEG, or WebP image.')

    return

  }



  if (file.size > 10 * 1024 * 1024) {

    setAnalysisError('Please select an image smaller than 10 MB.')

    return

  }



  if (imagePreview) URL.revokeObjectURL(imagePreview)



  setSelectedFile(file)

  setImagePreview(URL.createObjectURL(file))

  setImageName(file.name)

  setAnalysisError('')

  setResults(null)

}





async function analyzeImage() {

  if (!selectedFile || isAnalyzing) return



  setIsAnalyzing(true)

  setAnalysisError('')

  setResults(null)



  try {

    const formData = new FormData()

    formData.append('file', selectedFile)



    const response = await fetch('http://127.0.0.1:8000/predict', {

      method: 'POST',

      body: formData,

    })



    const data = await response.json()



    if (!response.ok) {

      throw new Error(data.detail || 'Image analysis failed.')

    }



    setResults(data)

  } catch (error) {

    setAnalysisError(

      error.message ||

      'Could not connect to the backend. Check that it is running.'

    )

  } finally {

    setIsAnalyzing(false)

  }

}






const navigation = [

    { name: 'Dashboard', icon: House },

    { name: 'Image Analysis', icon: FileImage },

    { name: 'Swede Score', icon: ClipboardList },

    { name: 'Reports', icon: ChartNoAxesCombined },

  ]



  return (

    <div className="app-shell">

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-icon">

            <Activity size={25} />

          </div>

          <div>

            <h2>CerviScan</h2>

            <p>AI-assisted research</p>

          </div>

        </div>



        <div className="nav-heading">WORKSPACE</div>



        <nav className="navigation">

          {navigation.map(({ name, icon: Icon }) => (

            <button

              key={name}

              className={`nav-item ${activePage === name ? 'active' : ''}`}

              onClick={() => setActivePage(name)}

            >

              <Icon size={19} />

              <span>{name}</span>

            </button>

          ))}

        </nav>



        <div className="sidebar-bottom">

          <div className="privacy-note">

            <ShieldCheck size={20} />

            <div>

              <strong>Research prototype</strong>

              <p>Not intended for clinical diagnosis.</p>

            </div>

          </div>



          <button

            className="nav-item"

            onClick={() => setActivePage('Settings')}

          >

            <Settings size={19} />

            <span>Settings</span>

          </button>



          <button

            className="nav-item"

            onClick={() => setActivePage('Help')}

          >

            <CircleHelp size={19} />

            <span>Help & information</span>

          </button>

        </div>

      </aside>



      <main className="main-content">

        <header className="topbar">

          <div>

            <span className="breadcrumb">Workspace /</span>

            <span className="breadcrumb-current">{activePage}</span>

          </div>

          <div className="topbar-right">

            <span className="status-dot" />

            <span className="system-status">Research interface</span>

            <div className="avatar">CS</div>

          </div>

        </header>



        <section className="page-content">

          <div className="welcome-row">

            <div>

              <div className="eyebrow">CERVICAL IMAGE ANALYSIS</div>

              <h1>

                {activePage === 'Dashboard'

                  ? 'Welcome to CerviScan'

                  : activePage}

              </h1>

              <p className="page-description">

                An AI-assisted research interface for colposcopy image

                analysis and explainability.

              </p>

            </div>

            <div className="date-label">

              <Brain size={18} />

              <span>EfficientNet-B0</span>

            </div>

          </div>



          <div className="notice">

            <ShieldCheck size={20} />

            <p>

              <strong>Research use only.</strong> Model predictions and

              visual explanations require expert review and must not be

              used as a standalone diagnosis.

            </p>

          </div>



          <div className="section-heading">

            <div>

              <h2>Image analysis</h2>

              <p>Select a colposcopy image from your device.</p>

            </div>

            <span className="step-label">STEP 01</span>

          </div>



          <div className="analysis-grid">

            <section className="panel upload-panel">

              <div className="panel-heading">

                <div className="panel-icon">

                  <FileImage size={21} />

                </div>

                <div>

                  <h3>Upload image</h3>

                  <p>Choose an image for analysis</p>

                </div>

              </div>



              <label className="upload-area">

                {imagePreview ? (

                  <img

                    src={imagePreview}

                    alt="Selected colposcopy preview"

                    className="image-preview"

                  />

                ) : (

                  <>

                    <div className="upload-icon">

                      <Upload size={28} />

                    </div>

                    <strong>Click to upload an image</strong>

                    <span>PNG, JPG, or JPEG</span>

                  </>

                )}



                <input

                  type="file"

                  accept="image/png,image/jpeg"

                  onChange={handleImageUpload}

                  hidden

                />

              </label>



              {imageName && (

                <div className="file-info">

                  <FileImage size={17} />

                  <span>{imageName}</span>

                  <button

                    type="button"

                    className="remove-image"

                    aria-label="Remove image"



                    onClick={() => {

                      if (imagePreview) URL.revokeObjectURL(imagePreview)



                        setImagePreview(null)

                        setImageName('')

                        setSelectedFile(null)

                        setResults(null)

                        setAnalysisError('')

                    }}



                  >

                    <X size={17} />

                  </button>

                </div>

              )}



              <button

                className="primary-button"

                disabled={!selectedFile || isAnalyzing}

                onClick={analyzeImage}

              >

                <Brain size={18} />

                {isAnalyzing ? 'Analysing image...' : 'Analyse image'}

              </button>



              <p className="helper-text">

                {isAnalyzing

                  ? 'Your image is being processed by the trained model.'

                  : 'Upload an image and run the research model.'}

              </p>

            </section>



            <section className="panel results-panel">

              <div className="panel-heading">

                <div className="panel-icon purple">

                  <Brain size={21} />

                </div>

                <div>

                  <h3>Prediction results</h3>

                  <p>Model output will appear here</p>

                </div>

              </div>





              {analysisError && (

                <div className="notice" role="alert">

                  <p>{analysisError}</p>

                </div>

              )}



              {results ? (

                <div className="prediction-results">

                  <div className="prediction-summary">

                    <p>Predicted class</p>

                    <h2>{results.predicted_class}</h2>

                    <p>

                      Model confidence:{' '}

                      {(results.confidence * 100).toFixed(2)}%

                    </p>

                  </div>



                  <h4>Class probabilities</h4>



                  <div className="class-list">

                    {Object.entries(results.probabilities).map(

                      ([label, probability]) => (

                        <div className="class-row" key={label}>

                          <span>{label}</span>

                          <span>

                            {(probability * 100).toFixed(2)}%

                          </span>

                        </div>

                      )

                    )}

                  </div>



                  <h4>Grad-CAM visualizations</h4>



                  <div className="gradcam-results">

                    {Object.entries(results.images).map(

                      ([label, imageUrl]) => (

                        <div className="gradcam-image" key={label}>

                          <p>

                            {label === 'original'

                              ? 'Original image'

                              : label === 'heatmap'

                                ? 'Grad-CAM heatmap'

                                : label === 'overlay'

                                  ? 'Grad-CAM overlay'

                                  : 'Annotated visualization'}

                          </p>



                          <img

                            src={imageUrl}

                            alt={label}

                            style={{

                              width: '100%',

                              maxWidth: '320px',

                              height: 'auto',

                              borderRadius: '10px',

                              objectFit: 'contain',

                            }}

                          />

                        </div>

                      )

                    )}

                  </div>



                  <p className="helper-text">

                    {results.disclaimer}

                  </p>

                </div>

              ) : (

                <div className="empty-state">

                  <div className="empty-illustration">

                    <Activity size={34} />

                  </div>



                  <h3>

                    {isAnalyzing

                      ? 'Analysing image...'

                      : 'Awaiting analysis'}

                  </h3>



                  <p>

                    {isAnalyzing

                      ? 'The model is processing your image.'

                      : 'Upload an image and click Analyse image to see the results.'}

                  </p>

                </div>

              )}



            </section>

          </div>

          {/* Supervisor's experimental Swede Score returned by backend.py */}
          {results?.swede_score && (
            <section className="panel swede-panel">
              <div className="panel-heading">
                <div className="panel-icon purple">
                  <ClipboardList size={21} />
                </div>
                <div>
                  <h3>Experimental Swede Score</h3>
                  <p>Calculated using the method implemented in predict.py</p>
                </div>
              </div>

              <div className="swede-total">
                <div>
                  <p>Calculated total</p>
                  <h2>
                    {results.swede_score.total}
                    <span> / {results.swede_score.maximum}</span>
                  </h2>
                  <p>
                    Grad-CAM lesion coverage:{' '}
                    {results.swede_score.lesion_coverage_percent}%
                  </p>
                </div>
                <div className="swede-risk">
                  <strong>{results.swede_score.risk_tag}</strong>
                  <p>Experimental category</p>
                </div>
              </div>

              <h4>Score breakdown</h4>
              <div className="swede-criteria">
                {[
                  { key: 'acetowhiteness', label: 'Acetowhiteness', source: 'Predicted-class lookup' },
                  { key: 'margins', label: 'Margins / surface', source: 'Predicted-class lookup' },
                  { key: 'vessels', label: 'Vessel patterns', source: 'Predicted-class lookup' },
                  { key: 'iodine', label: 'Iodine staining', source: 'Predicted-class lookup' },
                  { key: 'lesion_size', label: 'Lesion size', source: 'Grad-CAM estimate' },
                ].map((item) => (
                  <div className="swede-criterion" key={item.key}>
                    <div className="swede-criterion-info">
                      <strong>{item.label}</strong>
                      <p>{item.source}</p>
                    </div>
                    <strong>{results.swede_score[item.key]} / 2</strong>
                  </div>
                ))}
              </div>

              <div className="swede-risk-details">
                <h4>Category details</h4>
                <p>{results.swede_score.risk_level}</p>
              </div>

              <p className="helper-text">{results.swede_score.method}</p>
              <p className="helper-text">
                Research use only. This experimental heuristic is not a
                validated clinical Swede Score or a standalone clinical
                assessment. Four components are assigned from the predicted
                class; only lesion size is estimated from Grad-CAM coverage.
              </p>
            </section>
          )}

          



          <div className="section-heading lower-heading">

            <div>

              <h2>Explainability & clinical assessment</h2>

              <p>Research tools for reviewing the image.</p>

            </div>

          </div>



          <div className="feature-grid">

            <section className="feature-card">

              <div className="feature-icon teal">

                <Activity size={22} />

              </div>

              <h3>Grad-CAM explanation</h3>

              <p>

                View a heatmap of image regions influencing the model.

              </p>

              <span className="feature-status">

                Available in prediction results above

              </span>

            </section>



            <section className="feature-card">

              <div className="feature-icon blue">

                <ClipboardList size={22} />

              </div>

              <h3>Swede Score</h3>

              <p>

                Record five clinical criteria, each scored from 0 to 2,

                and calculate a total out of 10.

              </p>

              <span className="feature-status">

                Experimental score appears after image analysis

              </span>

            </section>



            <section className="feature-card">

              <div className="feature-icon purple">

                <Brain size={22} />

              </div>

              <h3>AI explanation assistant</h3>

              <p>

                A future feature for explaining research outputs and

                summarising results with appropriate limitations.

              </p>

              <span className="feature-status">Not connected</span>

            </section>

          </div>



          <footer className="page-footer">

            <span>CerviScan · AI-assisted research prototype</span>

            <span>For research use only</span>

          </footer>

        </section>

      </main>

    </div>

  )

}



export default App
