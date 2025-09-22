# 📊 Results

This folder contains the main evaluation figures from my Master’s thesis:  
*Enhancing Car-Like Robots’ Suspension Systems to Improve Performance and Load Safety With Support of Machine Learning.*

---

## How to Interpret
- The key metric shown here is **RMS body acceleration (m/s²)**.  
- **Lower RMS body acceleration = better ride comfort and payload safety.**  
- TD3 consistently shows lower RMS values compared to LQR and PID, especially on rougher road classes (D–E) and higher speeds.  

---

## Files
- **03_heatmaps.png** – RMS body acceleration heatmaps across speeds and road classes for each controller.  
- **05_summary.png** – Average performance comparison and active control benefits.  
- **06_road_profiles.png** – Example ISO 8608 road profiles (Classes A–E).  
- **15_boxplot_rms_A–E.png** – RMS distributions by road class, showing variance across trials for LQR, PID, and TD3.  
...
---

## More Information
For additional explanations, methodology, and detailed analysis, please refer to the full thesis:  
👉 [`thesis.pdf`](../Yagiz_Soyleguzel_thesis.pdf)
