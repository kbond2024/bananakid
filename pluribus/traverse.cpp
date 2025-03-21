#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <vector>
#include <pluribus/range.hpp>
#include <pluribus/mccfr.hpp>
#include <pluribus/poker.hpp>
#include <pluribus/cluster.hpp>
#include <pluribus/range_viewer.hpp>
#include <pluribus/util.hpp>
#include <pluribus/traverse.hpp>

namespace pluribus {

Action str_to_action(const std::string& str) {
  if(str.starts_with("check") || str.starts_with("call")) {
    return Action::CHECK_CALL;
  }
  else if(str.starts_with("fold")) {
    return Action::FOLD;
  }
  else if(str.starts_with("all-in")) {
    return Action::ALL_IN;
  }
  else if(str.starts_with("bet")) {
    return Action(std::stoi(str.substr(4)) / 100.0);
  }
  else {
    throw std::runtime_error("Invalid action string: " + str);
  }
}

std::unordered_map<Action, RenderableRange> trainer_ranges(const BlueprintTrainer& bp, const PokerState& state, 
                                                           const Board& board, PokerRange& base_range, bool force_regrets) {
  std::unordered_map<Action, RenderableRange> ranges;
  auto actions = valid_actions(state, bp.get_config().action_profile);
  auto color_map = map_colors(actions);
  for(Action a : actions) {
    PokerRange action_range;
    for(uint8_t i = 0; i < 52; ++i) {
      for(uint8_t j = i + 1; j < 52; ++j) {
        Hand hand{j, i};
        for(int card_idx = 0; card_idx < n_board_cards(state.get_round()); ++card_idx) {
          if(i == board.cards()[card_idx] || j == board.cards()[card_idx]) {
            base_range.set_frequency(hand, 0.0f);
          }
        }
        
        int cluster = FlatClusterMap::get_instance()->cluster(state.get_round(), board, hand);
        std::vector<float> freq;
        if(state.get_round() == 0 && !force_regrets) {
          int base_idx = bp.get_phi().index(state, cluster);
          freq = calculate_strategy(bp.get_phi(), base_idx, actions.size());
        }
        else {
          int base_idx = bp.get_regrets().index(state, cluster);
          freq = calculate_strategy(bp.get_regrets(), base_idx, actions.size());
        }
        int a_idx = std::distance(actions.begin(), std::find(actions.begin(), actions.end(), a));
        action_range.add_hand(hand, freq[a_idx]);
      }
    }
    ranges.insert({a, RenderableRange{base_range * action_range, a.to_string(), color_map[a], true}});
  }
  return ranges;
}

void render_ranges(RangeViewer* viewer_p, const PokerRange& base_range, const std::unordered_map<Action, RenderableRange>& action_ranges) {
  std::vector<RenderableRange> ranges{RenderableRange{base_range, "Base Range", Color{255, 255, 255, 255}}};
  for(auto& entry : action_ranges) ranges.push_back(entry.second);
  viewer_p->render(ranges);
}

void traverse(RangeViewer* viewer_p, const std::string& bp_fn) {
  std::cout << "Loading " << bp_fn << " for traversal... " << std::flush;
  auto bp = cereal_load<BlueprintTrainer>(bp_fn);
  std::cout << "Success.\n";

  std::string input;
  std::cout << "Board cards: ";
  auto board_cards = bp.get_config().init_board;
  for(uint8_t card : board_cards) std::cout << idx_to_card(card);
  if(board_cards.size() < 5) {
    std::getline(std::cin, input);
    uint8_t missing_cards[5];
    str_to_cards(input, missing_cards);
    for(int i = 0; i < 5 - bp.get_config().init_board.size(); ++i) {
      board_cards.push_back(missing_cards[i]);
    }
  }
  Board board(board_cards);
  std::cout << "Board: " << board.to_string() << "\n";

  PokerState state = bp.get_config().init_state;
  std::vector<PokerRange> ranges = bp.get_config().init_ranges;
  for(int i = 0; i < bp.get_config().poker.n_players; ++i) ranges.push_back(PokerRange::full());
  auto action_ranges = trainer_ranges(bp, state, board, ranges[state.get_active()], false);
  render_ranges(viewer_p, ranges[state.get_active()], action_ranges);

  std::cout << state.to_string();
  std::cout << "\nAction: ";
  while(std::getline(std::cin, input)) {
    if(input == "quit") {
      std::cout << "Exiting...\n\n";
      break;
    }
    else if(input == "reset") {
      std::cout << "Resetting...\n\n";
      ranges = bp.get_config().init_ranges;
      state = bp.get_config().init_state;
    }
    else {
      Action action = str_to_action(input);
      std::cout << "\n" << action.to_string() << "\n\n";
      ranges[state.get_active()] = action_ranges.at(action).get_range();
      state = state.apply(action);
    }
    
    if(state.is_terminal()) {
      ranges = bp.get_config().init_ranges;
      state = bp.get_config().init_state;
    }

    action_ranges = trainer_ranges(bp, state, board, ranges[state.get_active()], false);
    render_ranges(viewer_p, ranges[state.get_active()], action_ranges);
    std::cout << state.to_string();
    std::cout << "\nAction: ";
  }
}

}

