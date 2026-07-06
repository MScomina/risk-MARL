import sys
from typing import Iterable, Tuple

import pygame
import networkx as nx

from .constants import RiskPhase

# NOTE: This code has been created using AI (specifically the LLM gpt-oss:20b).
# This code is only responsible for rendering the board on screen, so i felt like
# it was ok to just let the LLM handle it.

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
DEFAULT_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

NODE_RADIUS = 26
EDGE_COLOR = (50, 50, 50)
EDGE_WIDTH = 2

FONT_SIZE = 14
ARMY_FONT_SZ = 24

AGENT_COLORS = [
    pygame.Color("dodgerblue"),
    pygame.Color("firebrick"),
    pygame.Color("forestgreen"),
    pygame.Color("goldenrod"),
    pygame.Color("orchid"),
    pygame.Color("steelblue"),
    pygame.Color("orange"),
    pygame.Color("mediumseagreen"),
]

CONTINENT_ALPHA = 50
CONTINENT_COLOR_CYCLE = [
    (255,   0,   0, CONTINENT_ALPHA),   # red
    (  0, 255,   0, CONTINENT_ALPHA),   # green
    (  0,   0, 255, CONTINENT_ALPHA),   # blue
    (255, 165,   0, CONTINENT_ALPHA),   # orange
    (128,   0,128, CONTINENT_ALPHA),    # purple
]

MARGIN_RATIO = 0.05

ARROW_COLOR = pygame.Color("yellow")
ARROW_SCALE = 1.0

class RiskPygameRenderer:
    """
    Draws the current state of a raw_env instance.
    """

    def __init__(self, env):
        self.env = env
        pygame.init()
        self.screen = pygame.display.set_mode(DEFAULT_WINDOW_SIZE)
        pygame.display.set_caption("Risk – PettingZoo")
        self.font = pygame.font.SysFont(None, FONT_SIZE)
        self.army_font = pygame.font.SysFont(None, ARMY_FONT_SZ)

        # Compute node positions once (layout is static)
        self.pos: dict[int, Tuple[float, float]] = self._compute_positions()
        self.paused = False

    def _draw_continents(self):
        """Draw a light‑transparent convex‑hull polygon for each continent."""
        overlay = pygame.Surface(DEFAULT_WINDOW_SIZE, flags=pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))

        color_cycle = CONTINENT_COLOR_CYCLE

        for i, (_, cont_data) in enumerate(self.env.map_network.graph["continents"].items()):
            pts = [self.pos[node] for node in cont_data["territories"] if node in self.pos]

            if len(pts) < 3:
                continue

            hull_pts = self._convex_hull(pts)

            pygame.draw.polygon(
                overlay,
                color_cycle[i % len(color_cycle)],
                hull_pts,
                width=0
            )

        self.screen.blit(overlay, (0, 0))

    def _convex_hull(self, pts: Iterable[Tuple[int, int]]) -> list:
        """
        Graham scan – returns the vertices of the convex hull in CCW order.
        `pts` is an iterable of (x, y) tuples.  The function works even if some points are duplicated.
        """
        pts = sorted(set(pts))
        if len(pts) <= 3:
            return pts

        def cross(o, a, b):
            """2‑D cross product of OA and OB vectors."""
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    def _compute_positions(self) -> dict[int, Tuple[float, float]]:
        G = self.env.map_network

        raw_pos = nx.spring_layout(G, seed=42, iterations=500, k=0.6)

        xs = [raw_pos[n][0] for n in G.nodes()]
        ys = [raw_pos[n][1] for n in G.nodes()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        width, height = DEFAULT_WINDOW_SIZE

        margin_w = MARGIN_RATIO * width
        margin_h = MARGIN_RATIO * height

        span_x = xmax - xmin or 1e-6
        span_y = ymax - ymin or 1e-6

        scale_x = (width - 2 * margin_w) / span_x
        scale_y = (height - 2 * margin_h) / span_y
        scale   = min(scale_x, scale_y)

        offset_x = margin_w - xmin * scale
        offset_y = margin_h - ymin * scale

        scaled = {
            n: (int(raw_pos[n][0] * scale + offset_x),
                int(raw_pos[n][1] * scale + offset_y))
            for n in G.nodes()
        }

        return scaled

    def _draw_edges(self):
        """Draw all edges as thin gray lines."""
        for u, v in self.env.map_network.edges():
            pygame.draw.line(
                self.screen,
                EDGE_COLOR,
                self.pos[u],
                self.pos[v],
                EDGE_WIDTH,
            )

    def _node_color(self, owner: int) -> pygame.Color:
        """
        Map an owning player index to a colour.  If the node is unowned
        (owner == -1) we use a neutral grey.
        """
        if owner < 0:
            return pygame.Color("gray")
        return AGENT_COLORS[owner % len(AGENT_COLORS)]

    def _draw_nodes(self):
        """Draw nodes, armies and optional text."""
        for node in self.env.map_network.nodes():
            idx = self.env.map_network.graph["node_to_idx"][node]
            owner = int(self.env.world_state["territory_owner"][idx])
            army  = int(self.env.world_state["number_of_armies"][idx])

            color = self._node_color(owner)

            pygame.draw.circle(
                self.screen,
                color,
                self.pos[node],
                NODE_RADIUS,
            )

            if army > 0:
                # Use the larger font for army counts
                txt = self.army_font.render(str(army), True, "black")
                txt_rect = txt.get_rect(center=self.pos[node])
                self.screen.blit(txt, txt_rect)

    def _draw_selected_node(self):
        """Draw a bright border around the currently selected node."""
        sel = int(self.env.world_state["selected_node"])
        if sel == -1:
            return

        pygame.draw.circle(
            self.screen,
            pygame.Color("white"),
            self.pos[self.env.map_network.graph["idx_to_node"][sel]],
            NODE_RADIUS + 4,
            width=3
        )

    def _draw_selected_edge(self):
        """Draw a bright line with an arrowhead over the currently selected edge."""
        sel = int(self.env.world_state["selected_edge"])
        if sel == -1:
            return

        u, v = self.env.map_network.graph["idx_to_edge"][sel]

        start_pos = pygame.math.Vector2(self.pos[u])
        end_pos   = pygame.math.Vector2(self.pos[v])

        pygame.draw.line(
            self.screen,
            ARROW_COLOR,
            (int(start_pos.x), int(start_pos.y)),
            (int(end_pos.x), int(end_pos.y)),
            width=4
        )

        direction = end_pos - start_pos

        if direction.length() == 0:
            return

        direction = direction.normalize()
        perp = pygame.math.Vector2(-direction.y, direction.x)

        arrow_size = NODE_RADIUS * ARROW_SCALE

        point1 = end_pos - direction * arrow_size + perp * (arrow_size / 2)
        point2 = end_pos - direction * arrow_size - perp * (arrow_size / 2)

        pygame.draw.polygon(
            self.screen,
            ARROW_COLOR,
            [
                (int(end_pos.x), int(end_pos.y)),
                (int(point1.x), int(point1.y)),
                (int(point2.x), int(point2.y))
            ]
        )

    def _draw_info(self):
        """Display current agent and phase."""
        cur_agent = self.env.agent_selection
        phase_obj = self.env.world_state["action_phase"]
        phase_name = RiskPhase(int(phase_obj)).name
        current_iter = self.env.num_moves

        txt1 = self.font.render(f"Agent: {cur_agent}", True, "white")
        txt2 = self.font.render(f"Phase: {phase_name}", True, "white")
        txt3 = self.font.render(f"Move number: {current_iter}", True, "white")

        self.screen.blit(txt1, (10, 10))
        self.screen.blit(txt2, (10, 30))
        self.screen.blit(txt3, (10, 50))

    def draw(self):
        """Render the current environment state to the window."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.key in (pygame.K_SPACE, pygame.K_p):
                    self.paused = not self.paused

        self.screen.fill((30, 30, 30))

        self._draw_continents()
        self._draw_edges()
        self._draw_selected_edge()
        self._draw_nodes()
        self._draw_selected_node()
        self._draw_info()

        pygame.display.flip()

    def quit(self):
        pygame.quit()