import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from openai import OpenAI
import re
from rag_helper import SimpleRAG
import os

# === LM Studio connection ===
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

def chat_with_model(prompt, model="local-model", max_tokens=750):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data assistant that predicts housing availability."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def extract_numbers(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return [float(n) for n in numbers]


# === Mapping of House Types to Applicant Types ===
HOUSE_TO_APPLICANT = {
    "Subsidized apartments / social housing": ["Family applicant"],
    "Studio apartment / shared rental": ["Individual"],
    "Senior housing / assisted living": ["Individual", "Family applicant"],
    "Dormitory / shared student housing": ["Individual", "Group"],
    "Shelter / transitional housing": ["Individual"],
    "Accessible housing / supported living": ["Individual", "Caregiver"],
    "Community housing / reserve housing": ["Group", "Band council"],
    "Temporary housing / rental unit": ["Family applicant", "Individual"],
    "Workforce housing / cooperative housing": ["Individual", "Group"],
    "Condominium / townhouse": ["Family applicant"],
    "Veteran housing / support program units": ["Individual", "Program"],
    "Crisis shelter / transitional housing": ["Individual", "Family applicant"],
    "Co-living or live–work unit": ["Individual"],
    "Farmworker housing / modular units": ["Group", "Employer"],
    "Condo / apartment rental": ["Individual", "Couple"],
    "Correctional facility / institutional unit": ["Government", "Justice system"],
    "Halfway house / supervised transitional unit": ["Individual", "Parole program"],
}

APPLICANT_TO_HOUSE = {}
for house, applicants in HOUSE_TO_APPLICANT.items():
    for a in applicants:
        APPLICANT_TO_HOUSE.setdefault(a, []).append(house)


# === Initialize RAG with relative paths ===
csv_files = [
    "housing_units_2.csv",
    "housing_units_3.csv",
    "applicants_2.csv",
    "applicants_3.csv"
]

rag = SimpleRAG(csv_files)


class HousePredictionApp(ctk.CTk):
    COLORS = ["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta", "olive", "pink"]

    def __init__(self):
        super().__init__()

        self.title("House Availability Prediction")
        self.geometry("1200x800")

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.submission_count = 0  # Tracks how many submissions so we assign colors

        # --- Top Frame ---
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(pady=10, padx=10, fill="x")

        applicant_label = ctk.CTkLabel(top_frame, text="Applicant:")
        applicant_label.pack(side="left", padx=5)

        applicant_values = ["-- Clear Selection --", "-- Select Applicant --"] + sorted(APPLICANT_TO_HOUSE.keys())
        self.applicant_option = ctk.CTkOptionMenu(top_frame, values=applicant_values, command=self.on_applicant_selected)
        self.applicant_option.pack(side="left", padx=10)
        self.applicant_option.set("-- Select Applicant --")

        house_label = ctk.CTkLabel(top_frame, text="House Type:")
        house_label.pack(side="left", padx=5)

        house_values = ["-- Clear Selection --", "-- Select House Type --"] + sorted(HOUSE_TO_APPLICANT.keys())
        self.house_option = ctk.CTkOptionMenu(top_frame, values=house_values, command=self.on_house_selected)
        self.house_option.pack(side="left", padx=10)
        self.house_option.set("-- Select House Type --")

        # --- Graph Area ---
        mid_frame = ctk.CTkFrame(self)
        mid_frame.pack(pady=20, padx=10, fill="both", expand=True)

        self.graph_canvases = []
        self.graph_axes = []
        self.history_data = [[], [], []]  # store all previous submissions for 3 graphs

        # Graphs
        self.create_graph(mid_frame, 0, "Placed", "Time", "Houses", y_lim=(0, 10))
        self.create_graph(mid_frame, 1, "Need Place", "Time", "Applicants", y_lim=(0, 10))
        self.create_graph(mid_frame, 2, "Difference", "Time", "Housing Available - Applicants", y_lim=(-10, 10))

        # --- Bottom Frame ---
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(pady=10, padx=10, fill="x")

        self.submit_button = ctk.CTkButton(bottom_frame, text="Submit", command=self.submit_action)
        self.submit_button.pack(side="left", padx=10, pady=10)

        self.time_slider = ctk.CTkSlider(bottom_frame, from_=0, to=100, command=self.update_time_label)
        self.time_slider.pack(side="left", fill="x", expand=True, padx=(20, 10))

        self.time_label = ctk.CTkLabel(self, text="Day 1", anchor="e", justify="right")
        self.time_label.pack(anchor="e", padx=20, pady=(0, 10))

    # --- Graph creation ---
    def create_graph(self, parent, column, title, xlabel, ylabel, y_lim=(0, 10)):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=0, column=column, padx=10, pady=10, sticky="nsew")
        parent.grid_columnconfigure(column, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        fig = Figure(figsize=(3, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot([0], [0], color="gray")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(y_lim)
        ax.set_xlim(0, 10)
        ax.grid(True, linestyle="--", alpha=0.5)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self.graph_canvases.append(canvas)
        self.graph_axes.append(ax)

    # --- Dropdown logic ---
    def on_house_selected(self, selected):
        if selected == "-- Clear Selection --":
            self.reset_dropdowns()
            return
        if selected.startswith("--"):
            return
        applicants = HOUSE_TO_APPLICANT.get(selected, [])
        if applicants:
            new_values = ["-- Clear Selection --"] + applicants
            self.applicant_option.configure(values=new_values)
            self.applicant_option.set(applicants[0])

    def on_applicant_selected(self, selected):
        if selected == "-- Clear Selection --":
            self.reset_dropdowns()
            return
        if selected.startswith("--"):
            return
        houses = APPLICANT_TO_HOUSE.get(selected, [])
        if houses:
            new_values = ["-- Clear Selection --"] + houses
            self.house_option.configure(values=new_values)
            self.house_option.set(houses[0])

    def reset_dropdowns(self):
        self.applicant_option.configure(values=["-- Clear Selection --", "-- Select Applicant --"] + sorted(APPLICANT_TO_HOUSE.keys()))
        self.applicant_option.set("-- Select Applicant --")
        self.house_option.configure(values=["-- Clear Selection --", "-- Select House Type --"] + sorted(HOUSE_TO_APPLICANT.keys()))
        self.house_option.set("-- Select House Type --")

    # --- Slider label / units ---
    def update_time_label(self, value):
        v = float(value)
        # dynamically adjust unit
        if v < 10:
            label = "Hour 1"
        elif v < 20:
            label = "Hour 2"
        elif v < 30:
            label = "Day 1"
        elif v < 40:
            label = "Day 2"
        elif v < 50:
            label = "Week 1"
        elif v < 60:
            label = "Week 2"
        elif v < 70:
            label = "Month 1"
        elif v < 80:
            label = "Month 2"
        elif v < 90:
            label = "Year 1"
        else:
            label = "Year 5"
        self.time_label.configure(text=label)

    # --- Submit Action ---
    def submit_action(self):
        applicant = self.applicant_option.get()
        house = self.house_option.get()
        time_label = self.time_label.cget("text")

        # retrieve RAG context
        query = f"Applicant: {applicant}, House: {house}, Time: {time_label}"
        context = rag.build_context(query)

        prompt = f"""
        Predict housing availability for:
        Applicant type: {applicant}
        House type: {house}
        Time frame: {time_label}

        Include relevant historical data:

        {context}
        """

        try:
            reply = chat_with_model(prompt)
            print("\nModel reply:\n", reply)

            numbers = extract_numbers(reply)
            if not numbers:
                print("No numeric data found in model response.")
                return

            # split numbers for 3 graphs
            chunk = len(numbers) // 3 or 1
            placed = numbers[:chunk]
            need = numbers[chunk:2*chunk]
            diff = numbers[2*chunk:] or [p - n for p, n in zip(placed, need)]

            data_sets = [placed, need, diff]

            # --- Update graphs with persistent lines ---
            color = self.COLORS[self.submission_count % len(self.COLORS)]
            self.submission_count += 1

            for i, (ax, canvas, data) in enumerate(zip(self.graph_axes, self.graph_canvases, data_sets)):
                # store history
                self.history_data[i].append((data, color))

                ax.clear()
                # plot all previous submissions
                for past_data, past_color in self.history_data[i]:
                    ax.plot(range(len(past_data)), past_data, marker="o", linestyle="-", color=past_color)

                ax.set_title(["Placed", "Need Place", "Difference"][i])
                ax.set_xlabel("Time")
                ax.set_ylabel(["Houses", "Applicants", "Housing Available - Applicants"][i])
                ax.grid(True, linestyle="--", alpha=0.5)

                # dynamic Y-axis
                if i == 2:  # difference can be negative
                    all_values = [v for series, _ in self.history_data[i] for v in series]
                    max_val = max(abs(min(all_values)), max(all_values)) * 1.2
                    ax.set_ylim(-max_val, max_val)
                else:
                    all_values = [v for series, _ in self.history_data[i] for v in series]
                    ax.set_ylim(0, max(all_values)*1.2 if max(all_values) > 0 else 10)

                # X-axis units
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels([time_label]*len(data), rotation=45, fontsize=8)

                canvas.draw()

        except Exception as e:
            print("Error communicating with LM Studio:", e)


if __name__ == "__main__":
    app = HousePredictionApp()
    app.mainloop()
