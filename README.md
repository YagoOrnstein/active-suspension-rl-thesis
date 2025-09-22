# active-suspension-rl-thesis
Master’s thesis project: Reinforcement Learning (TD3) for active suspension control in car-like robots. Includes nonlinear quarter-car model, ISO 8608 road profiles, and comparison with LQR &amp; PID.

# Enhancing Car-Like Robots’ Suspension Systems with Reinforcement Learning

This repository contains the code, experiments, and thesis from my **Master’s degree in Robotics** at *Warsaw University of Technology*.  
The research focuses on applying **Reinforcement Learning (TD3)** to improve active suspension control in car-like robotic platforms.  

---

## 📝 Thesis
- **Title:** Enhancing Car-Like Robots’ Suspension Systems to Improve Performance and Load Safety With Support of Machine Learning  
- **Author:** Yağız Söylegüzel  
- **Institution:** Warsaw University of Technology (Politechnika Warszawska)  
- **Year:** 2025  
- **Keywords:** Reinforcement Learning, TD3, Active Suspension, Vehicle Dynamics, Robotics, Control Systems  

📄 Full thesis: [`thesis.pdf`](Yagiz_Soyleguzel_thesis.pdf)

---

## ⚙️ Project Overview
The project addresses suspension control for car-like robots operating in diverse terrain conditions. Traditional controllers (LQR, PID) face challenges under nonlinearities, varying loads, and rough road profiles.  

This work develops:  
1. A **nonlinear quarter-car model** that includes:  
   - Progressive springs  
   - Hysteretic friction  
   - Adaptive damping  
   - Load variation  
   - Speed-dependent tire stiffness  
   - Actuator force saturation  

2. A **TD3 reinforcement learning controller**, trained with:  
   - Curriculum learning (increasing speeds, roughness)  
   - LQR demonstrations for stability guidance  
   - Safety filters and rollback strategies  

3. A **comprehensive evaluation** against passive suspension, PID, and LQR controllers.  

---

## 📊 Results
- TD3 **outperforms LQR and PID** across ISO 8608 road classes (C–E) and speeds (25–55 km/h).  
- Significant improvements in **ride comfort (lower RMS body acceleration)** and **payload safety**.  
- More robust performance in **harsh road conditions** and **higher vehicle speeds**.  

Example plots (from `results/` folder):  
- RMS acceleration vs. road class  
- TD3 vs LQR performance gaps  
- Frequency and step response validation  
- Energy balance and nonlinear effect analysis  

---

## 📂 Repository Contents
- `Yagiz_Soyleguzel_thesis.pdf` – full thesis document  
- `td3_active_suspension_nonlinear.py` – nonlinear quarter-car environment & TD3 training script  
- `evaluations.py` – evaluation and plotting framework  
- `results/` – generated figures, tables, and analysis plots  
- `README.md` – this documentation  

---

## 🛠️ Tech Stack
- **Languages:** Python,   
- **Libraries:** PyTorch, NumPy, Matplotlib, SciPy  
- **Domains:** Reinforcement Learning, Vehicle Dynamics, Control Systems  

---
