# 🎼 SpikeSymphony

**SpikeSymphony** presents two interactive, event-driven music demos — **Spiky Piano** and **SpecDrum** — powered by neuromorphic vision. Using an event-based camera (DVS), these systems detect spatial activity and transform it into musical output in real time.

> 📌 **Submission for [SpikeCV Competition – Track 3: Event-Based Creative Demos](https://spikecv.github.io/competition)**

---

## 🎹 Included Demos

### 🔸 [Spiky Piano](./Spiky_Piano/)

- Detects object **size** using connected components on event streams
- Categorises blobs into **small**, **medium**, or **large**
- Each size triggers a corresponding **musical note**
- Calibrated visual overlay with counters, thresholds, and interactive reset

→ Explore the full description and run instructions in the [Spiky Piano README](./Spiky_Piano/README.md)

---

### 🔸 [SpecDrum](./SpecDrum/)

- Divides the visual field into **four vertical zones**
- Monitors vibrational activity via **frequency bursts**
- Triggers one of **four drum samples** depending on active region
- Live heatmap GUI to visualize spatial energy patterns

→ Full details and parameter guide available in the [SpecDrum README](./SpecDrum/README.md)

---

## 🛠️ Technologies Used

- **Event Camera**: Prophesee SilkyEvCam
- **SDK**: Metavision SDK (Core, CV, Analytics, UI)
- **Language**: Python 3.8+
- **Libraries**: `opencv-python`, `pygame`, `numpy`, `metavision-sdk`

---

## 👥 Team

- **Team Lead**: Muhammad Aitsam  
  Marie Skłodowska-Curie Fellow
  Researcher/ PhD Candidate Sheffield Hallam University, UK
  [Website](https://sites.google.com/view/aitsam) • [GitHub](https://github.com/aitsam12)

- **Technical Support**: Syed Saad Hassan
  Sapienza University of Rome, Italy

- **Supervision**: Dr. Alejandro Jiménez Rodríguez  
  Senior Lecturer, Sheffield Hallam University, UK

---

## 🏆 Competition Context

This project is submitted to the **SpikeCV 2025 Competition**, specifically:

> **Track 3: Creative Applications and Demos**  
> https://spikecv.github.io/competition

SpikeSymphony highlights the potential of neuromorphic cameras in **interactive arts**, **HCI**, and **musical computing**, demonstrating real-world applications of spatial and frequency-aware processing on asynchronous vision data.

---

## 📜 License

This repository is distributed under the **MIT License**. See [LICENSE](./LICENSE) for full terms.

---

## 🔗 Related Links

- [Spiky Piano Demo →](./Spiky_Piano/)
- [SpecDrum Demo →](./SpecDrum/)
- [SpikeCV Competition Website →](https://spikecv.github.io/competition)

