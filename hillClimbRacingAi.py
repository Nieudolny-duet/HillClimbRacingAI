import numpy as np
import cv2
import pyautogui
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
from collections import deque
import os
import ctypes
import psutil
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import gc
import logging
import winsound
import keyboard
from flask import Flask, render_template_string
import socket
from queue import Queue
 
# Konfiguracja loggera
logging.basicConfig(filename='ai_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
 
print("Program wystartował")
 
# Stałe
MONEY_ADDRESS = 0x97D6D278
PLAYER_ALIVE_ADDRESS = 0x032543D0
DISTANCE_ADDRESS = 0x804EA5CC
FRAME_TIME = 1/40  # FPS
FAIL_CLICK_POS = (1552, 781)
WAIT_TIME = 5
START_KEY = 'f10'
MAX_IMAGES = 10001
 
# Globalne zmienne do przechowywania statystyk
total_money = 0
total_distance = 0
episode_count = 0
message_queue = Queue()
 
# Inicjalizacja aplikacji Flask
app = Flask(__name__)
 
@app.route('/')
def home():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
 
    messages = []
    while not message_queue.empty():
        messages.append(message_queue.get())
 
    html = '''
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hill Climb Racing AI - Statystyki</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; }
            .stat { margin: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            #messages { max-height: 300px; overflow-y: auto; text-align: left; margin: 20px; padding: 10px; border: 1px solid #ddd; }
        </style>
        <script>
            function updateStats() {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('total_money').textContent = data.total_money;
                        document.getElementById('total_distance').textContent = data.total_distance;
                        document.getElementById('episode_count').textContent = data.episode_count;
                        document.getElementById('player_alive').textContent = data.player_alive;
                        document.getElementById('current_money').textContent = data.current_money;
                    });
 
                fetch('/messages')
                    .then(response => response.json())
                    .then(data => {
                        const messagesDiv = document.getElementById('messages');
                        data.forEach(message => {
                            const p = document.createElement('p');
                            p.textContent = message;
                            messagesDiv.appendChild(p);
                        });
                        messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    });
            }
 
            setInterval(updateStats, 1000);
        </script>
    </head>
    <body>
        <h1>Hill Climb Racing AI - Statystyki</h1>
        <div class="stat">
            <h2>Użycie CPU: {{ cpu_percent }}%</h2>
        </div>
        <div class="stat">
            <h2>Użycie pamięci: {{ memory_percent }}%</h2>
        </div>
        <div class="stat">
            <h2>Użycie dysku: {{ disk_percent }}%</h2>
        </div>
        <div class="stat">
            <h2>Całkowita ilość pieniędzy: <span id="total_money">{{ total_money }}</span></h2>
        </div>
        <div class="stat">
            <h2>Całkowity przebyty dystans: <span id="total_distance">{{ total_distance }}</span></h2>
        </div>
        <div class="stat">
            <h2>Liczba epizodów: <span id="episode_count">{{ episode_count }}</span></h2>
        </div>
        <div class="stat">
            <h2>Stan gracza (0x0D560B78): <span id="player_alive">{{ player_alive }}</span></h2>
        </div>
        <div class="stat">
            <h2>Aktualne pieniądze (0x8CA8C278): <span id="current_money">{{ current_money }}</span></h2>
        </div>
        <div id="messages"></div>
    </body>
    </html>
    '''
    return render_template_string(html, cpu_percent=cpu_percent, memory_percent=memory_percent, disk_percent=disk_percent,
                                  total_money=total_money, total_distance=total_distance, episode_count=episode_count,
                                  player_alive=env.read_memory(PLAYER_ALIVE_ADDRESS),
                                  current_money=env.read_memory(MONEY_ADDRESS))
 
@app.route('/stats')
def stats():
    return {
        'total_money': total_money,
        'total_distance': total_distance,
        'episode_count': episode_count,
        'player_alive': env.read_memory(PLAYER_ALIVE_ADDRESS),
        'current_money': env.read_memory(MONEY_ADDRESS)
    }
 
@app.route('/messages')
def messages():
    messages = []
    while not message_queue.empty():
        messages.append(message_queue.get())
    return messages
 
def run_flask():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print(f"Serwer uruchomiony. Otwórz http://{ip_address}:5000 w przeglądarce na innym komputerze w sieci lokalnej.")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
 
class ButtonController(threading.Thread):
    def __init__(self, accelerate_pos, brake_pos):
        threading.Thread.__init__(self)
        self.accelerate_pos = accelerate_pos
        self.brake_pos = brake_pos
        self.action = None
        self.running = False
        self.paused = True
        self.short_action_duration = 0.1  # Czas trwania krótkiej akcji (100ms)
        self.long_action_duration = 0.5  # Czas trwania długiej akcji (500ms)
 
    def run(self):
        while self.running:
            if not self.paused and self.action is not None:
                if self.action == 'short_accelerate':
                    pyautogui.mouseDown(button='left', x=self.accelerate_pos[0], y=self.accelerate_pos[1])
                    time.sleep(self.short_action_duration)
                    pyautogui.mouseUp(button='left', x=self.accelerate_pos[0], y=self.accelerate_pos[1])
                elif self.action == 'long_accelerate':
                    pyautogui.mouseDown(button='left', x=self.accelerate_pos[0], y=self.accelerate_pos[1])
                    time.sleep(self.long_action_duration)
                    pyautogui.mouseUp(button='left', x=self.accelerate_pos[0], y=self.accelerate_pos[1])
                elif self.action == 'brake':
                    pyautogui.mouseDown(button='left', x=self.brake_pos[0], y=self.brake_pos[1])
                    time.sleep(self.short_action_duration)
                    pyautogui.mouseUp(button='left', x=self.brake_pos[0], y=self.brake_pos[1])
                self.action = None
            time.sleep(0.01)
 
    def set_action(self, action):
        self.action = action
 
    def start_controller(self):
        self.running = True
        self.paused = False
        if not self.is_alive():
            self.start()
 
    def pause(self):
        self.paused = True
 
    def resume(self):
        self.paused = False
 
    def stop(self):
        self.running = False
        self.join()
 
class HillClimbEnv:
    def __init__(self, reward_type):
        self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        self.frame_count = 0
 
        self.screenshot_folder = "screenshots"
        self.gray_folder = os.path.join(self.screenshot_folder, "gray")
        self.edges_folder = os.path.join(self.screenshot_folder, "edges")
        os.makedirs(self.gray_folder, exist_ok=True)
        os.makedirs(self.edges_folder, exist_ok=True)
 
        self.accelerate_pos = (1721, 870)
        self.brake_pos = (227, 890)
 
        self.fail_click_pos = FAIL_CLICK_POS
 
        self.process_handle = self.open_process()
        self.current_money = self.read_memory(MONEY_ADDRESS)
        self.player_alive = self.read_memory(PLAYER_ALIVE_ADDRESS)
        self.current_distance = self.read_memory(DISTANCE_ADDRESS)
        self.last_distance = self.current_distance
        self.last_money = self.current_money
 
        self.button_controller = ButtonController(self.accelerate_pos, self.brake_pos)
        self.fail_cooldown = 0
        self.error_count = 0
        self.max_errors = 5
 
        self.start_time = time.time()
        self.reward_type = reward_type
 
        self.game_state = "menu"
        self.min_episode_duration = 30  # Minimalny czas trwania epizodu w sekundach
        self.episode_start_time = time.time()
 
    def start_controller(self):
        self.button_controller.start_controller()
 
    def open_process(self):
        game_process_name = "vboxheadless.exe"
        pid = self.get_pid_by_name(game_process_name)
        if pid is None:
            raise Exception(f"Nie znaleziono procesu o nazwie {game_process_name}")
        h_process = ctypes.windll.kernel32.OpenProcess(0x10, False, pid)
        return h_process
 
    def get_pid_by_name(self, process_name):
        for proc in psutil.process_iter(['name', 'pid']):
            if proc.info['name'].lower() == process_name.lower():
                return proc.info['pid']
        return None
 
    def read_memory(self, address):
        buffer = ctypes.c_uint32()
        bytes_read = ctypes.c_uint32()
        ctypes.windll.kernel32.ReadProcessMemory(self.process_handle, address, ctypes.byref(buffer), ctypes.sizeof(buffer), ctypes.byref(bytes_read))
        return buffer.value
 
    def reset(self):
        global total_distance, total_money, episode_count
        self.button_controller.set_action(None)
        time.sleep(1)
        pyautogui.click(x=960, y=540)
        time.sleep(1)
        self.start_time = time.time()
        self.current_distance = self.read_memory(DISTANCE_ADDRESS)
        self.last_distance = self.current_distance
        self.current_money = self.read_memory(MONEY_ADDRESS)
        self.last_money = self.current_money
        episode_count += 1
        message = f"Epizod {episode_count} rozpoczęty. Całkowity dystans: {total_distance}, Całkowite pieniądze: {total_money}"
        print(message)
        message_queue.put(message)
        self.game_state = "playing"
        print("Gra rozpoczęta - AI nie uczy się podczas jazdy")
        self.episode_start_time = time.time()
        time.sleep(2)  # Opóźnienie przed rozpoczęciem nowego epizodu
        return self._get_state()
 
    def step(self, action):
        try:
            start_time = time.time()
 
            if self.game_state == "playing":
                if action == 0:  # Krótkie przyspieszenie
                    self.button_controller.set_action('short_accelerate')
                elif action == 1:  # Długie przyspieszenie
                    self.button_controller.set_action('long_accelerate')
                elif action == 2:  # Hamowanie
                    self.button_controller.set_action('brake')
                else:  # Nic nie rób
                    self.button_controller.set_action(None)
 
            new_state = self._get_state()
            reward = self._calculate_reward()
            done = self._is_fail()
 
            if done:
                if time.time() - self.episode_start_time < self.min_episode_duration:
                    reward -= 50000  # Duża kara za zbyt szybką śmierć
                else:
                    reward -= 10000  # Standardowa kara za śmierć
                self.button_controller.set_action(None)
                self.handle_fail()
                self.game_state = "menu"
                print("Gra zakończona - AI uczy się w menu")
                time.sleep(2)  # Dodatkowe opóźnienie po śmierci
 
            elapsed_time = time.time() - start_time
            if elapsed_time < FRAME_TIME:
                time.sleep(FRAME_TIME - elapsed_time)
 
            return new_state, reward, done
        except Exception as e:
            logging.error(f"Error in step: {e}")
            return None, -50, True
 
    def _get_state(self):
        try:
            self.frame_count += 1
 
            screenshot = pyautogui.screenshot(region=(self.monitor['left'], self.monitor['top'], 
                                                      self.monitor['width'], self.monitor['height']))
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
 
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_edges = cv2.Canny(img_gray, 100, 200)
 
            resized_gray = cv2.resize(img_gray, (256, 256))
            resized_edges = cv2.resize(img_edges, (256, 256))
 
            frame_index = self.frame_count % MAX_IMAGES
            cv2.imwrite(os.path.join(self.gray_folder, f"frame_{frame_index:05d}.png"), resized_gray)
            cv2.imwrite(os.path.join(self.edges_folder, f"frame_{frame_index:05d}.png"), resized_edges)
 
            self.current_money = self.read_memory(MONEY_ADDRESS)
            self.player_alive = self.read_memory(PLAYER_ALIVE_ADDRESS)
            self.current_distance = self.read_memory(DISTANCE_ADDRESS)
 
            if self.frame_count % 100 == 0:
                gc.collect()
 
            player_state = np.full((256, 256), self.player_alive)
            distance_state = np.full((256, 256), self.current_distance)
            state = np.stack([resized_edges, player_state, distance_state], axis=0)
 
            return state
        except Exception as e:
            logging.error(f"Error in _get_state: {e}")
            self.error_count += 1
            if self.error_count > self.max_errors:
                raise Exception("Too many consecutive errors")
            return None
 
    def _calculate_reward(self):
        global total_money, total_distance
        reward = 0
 
        if self.reward_type == 'distance':
            distance_gained = self.current_distance - self.last_distance
            if distance_gained > 0:
                reward += distance_gained  # Zwiększona nagroda za dystans
                total_distance += distance_gained
            self.last_distance = self.current_distance
        elif self.reward_type == 'money':
            money_gained = self.current_money - self.last_money
            if money_gained > 0:
                reward += money_gained * 10  # Zwiększona nagroda za pieniądze
                total_money += money_gained
                message = f"Zdobyto {money_gained} pieniędzy! Łącznie: {total_money}"
                print(message)
                message_queue.put(message)
            self.last_money = self.current_money
 
        survival_time = time.time() - self.start_time
        reward += survival_time * 0.1  # Znacznie zwiększona nagroda za czas przeżycia
 
        return reward
 
    def _is_fail(self):
        return self.player_alive == 0 or self.current_distance < self.last_distance
 
    def handle_fail(self):
        pyautogui.click(x=self.fail_click_pos[0], y=self.fail_click_pos[1])
        winsound.Beep(1000, 500)
        time.sleep(0.5)
        pyautogui.click(x=self.fail_click_pos[0], y=self.fail_click_pos[1])
 
    def __del__(self):
        if hasattr(self, 'button_controller'):
            self.button_controller.stop()
        if hasattr(self, 'process_handle'):
            ctypes.windll.kernel32.CloseHandle(self.process_handle)
 
class ImprovedActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ImprovedActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
 
        self.lin1 = nn.Linear(self.feature_size(), 512)
        self.lin2 = nn.Linear(512, 256)
 
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)
 
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
 
        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic
 
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, 3, 256, 256)))).view(1, -1).size(1)
 
class ImprovedPPOAgent:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.n_actions = 4  # Zmieniono na 4 akcje (krótkie przyspieszenie, długie przyspieszenie, hamowanie, nic nie rób)
        self.model = ImprovedActorCritic(3, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.1
        self.training_buffer = deque(maxlen=100000)
 
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1), None
 
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        dist = Categorical(torch.softmax(action_probs, dim=-1))
        action = dist.sample()
        return action.item(), dist.log_prob(action)
 
    def update(self):
        if len(self.training_buffer) < self.batch_size * 10:
            return
 
        batch = random.sample(self.training_buffer, self.batch_size)
        states, actions, old_probs, rewards, next_states, dones = zip(*batch)
 
        states = torch.stack([torch.from_numpy(state).float() for state in states]).to(self.device)
        next_states = torch.stack([torch.from_numpy(state).float() for state in next_states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_probs = torch.stack([prob if prob is not None else torch.zeros(1).to(self.device) for prob in old_probs]).to(self.device).detach()
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
 
        for _ in range(self.K_epochs):
            action_probs, state_values = self.model(states)
            _, next_state_values = self.model(next_states)
 
            dist = Categorical(torch.softmax(action_probs, dim=-1))
 
            advantages = rewards + self.gamma * next_state_values.squeeze() * (1 - dones) - state_values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
            ratios = torch.exp(dist.log_prob(actions) - old_probs)
 
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
 
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), rewards + self.gamma * next_state_values.squeeze() * (1 - dones))
            entropy = dist.entropy().mean()
 
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
 
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
 
    def train(self, num_episodes):
        global episode_count, total_money, total_distance
        for i_episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
 
            while True:
                action, log_prob = self.select_action(state)
                next_state, reward, done = self.env.step(action)
 
                if next_state is None:
                    break
 
                if self.env.game_state == "menu":
                    self.training_buffer.append((state, action, log_prob, reward, next_state, done))
                    self.update()
 
                state = next_state
                episode_reward += reward
 
                if done:
                    break
 
            message = f"Epizod {i_episode + 1} zakończony z nagrodą {episode_reward:.2f}. Łącznie pieniądze: {total_money}, dystans: {total_distance}"
            print(message)
            logging.info(message)
            message_queue.put(message)
 
            if (i_episode + 1) % 10 == 0:
                self.save_model(i_episode + 1)
 
            if (i_episode + 1) % 50 == 0:
                gc.collect()
 
    def save_model(self, episode):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': episode,
        }, os.path.join(self.env.screenshot_folder, f"improved_model_episode_{episode}.pth"))
        message = f"Model zapisany po epizodzie {episode}"
        print(message)
        logging.info(message)
        message_queue.put(message)
 
    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode']
            message = f"Model wczytany z epizodu {start_episode}"
            print(message)
            logging.info(message)
            message_queue.put(message)
            return start_episode
        else:
            message = "Nie znaleziono zapisanego modelu. Rozpoczynanie od nowa."
            print(message)
            logging.info(message)
            message_queue.put(message)
            return 0
 
def create_start_window():
    window = tk.Tk()
    window.title("Hill Climb Racing AI")
    window.geometry("300x200")
 
    reward_type = tk.StringVar(value="distance")
 
    def start_new_training():
        window.destroy()
        main(None, reward_type.get())
 
    def load_existing_model():
        file_path = filedialog.askopenfilename(initialdir="./screenshots", 
                                               title="Select model file", 
                                               filetypes=(("PTH files", "*.pth"), ("all files", "*.*")))
        if file_path:
            window.destroy()
            main(file_path, reward_type.get())
        else:
            messagebox.showinfo("Info", "Nie wybrano pliku. Rozpoczynanie nowego treningu.")
            window.destroy()
            main(None, reward_type.get())
 
    tk.Label(window, text="Wybierz typ nagrody:").pack(pady=10)
    tk.Radiobutton(window, text="Dystans", variable=reward_type, value="distance").pack()
    tk.Radiobutton(window, text="Pieniądze", variable=reward_type, value="money").pack()
 
    tk.Button(window, text="Rozpocznij nowy trening", command=start_new_training).pack(pady=10)
    tk.Button(window, text="Wczytaj istniejący model", command=load_existing_model).pack(pady=10)
 
    window.mainloop()
 
def main(load_model_path, reward_type):
    global env
    env = HillClimbEnv(reward_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ImprovedPPOAgent(env, device)
 
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
 
    try:
        print(f"Naciśnij '{START_KEY}' aby rozpocząć trening...")
        keyboard.wait(START_KEY)
        print("Klawisz START naciśnięty. Rozpoczynam trening...")
        message = f"Trenowanie modelu z nagrodą za: {reward_type}"
        print(message)
        message_queue.put(message)
 
        if load_model_path:
            agent.load_model(load_model_path)
 
        env.start_controller()
        agent.train(1000)
    except Exception as e:
        logging.error(f"Główny błąd: {e}")
        print(f"Wystąpił błąd: {e}")
        print("Program zostaje zatrzymany.")
    finally:
        if hasattr(env, 'button_controller'):
            env.button_controller.stop()
        gc.collect()
 
if __name__ == "__main__":
    create_start_window()