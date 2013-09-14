/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package util.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 *
 * @author vietan
 */
public class Conversation {

    protected final String id;
    protected ArrayList<Turn> turns;
    protected Set<String> speakerSet;

    public Conversation(String id) {
        this.id = id;
        this.turns = new ArrayList<Turn>();
        this.speakerSet = new HashSet<String>();
    }

    public String getID() {
        return this.id;
    }

    public int getNumTurns() {
        return this.turns.size();
    }

    public int getNumSpeakers() {
        return this.speakerSet.size();
    }

    public void addTurn(Turn t) {
        this.turns.add(t);
        String speaker = t.getSpeaker();
        speakerSet.add(speaker);
    }

    public Turn getTurn(int i) {
        return this.turns.get(i);
    }

    public ArrayList<Turn> getTurns() {
        return this.turns;
    }

    public void setSpeakers(Set<String> speakers) {
        this.speakerSet = speakers;
    }

    public Set<String> getSpeakers() {
        return this.speakerSet;
    }

    public void resolveSpeakers() {
        HashMap<String, String> name_map = new HashMap<String, String>();
        for (String speaker_name : speakerSet) {
            for (String other_name : speakerSet) {
                if (!speaker_name.equals(other_name)
                        && speaker_name.contains(other_name)) {
                    name_map.put(speaker_name, other_name);
                }
            }
        }

        this.speakerSet = new HashSet<String>();
        for (Turn turn : turns) {
            String speaker = turn.getSpeaker();
            if (name_map.get(speaker) != null) {
                turn.setSpeaker(name_map.get(speaker));
            }
            this.speakerSet.add(turn.getSpeaker());
        }
    }

    public void resolveSplitTurns() {
        ArrayList<Turn> tempTurns = new ArrayList<Turn>();
        tempTurns.add(this.getTurn(0));
        Turn preTurn, posTurn;
        for (int i = 1; i < this.turns.size(); i++) {
            preTurn = this.turns.get(i - 1);
            posTurn = this.turns.get(i);
            if (preTurn.getSpeaker().equals(posTurn.getSpeaker())) {
                preTurn.setText(preTurn.getText() + " " + posTurn.getText());
            } else {
                tempTurns.add(posTurn);
            }
        }

        for (int i = 0; i < tempTurns.size(); i++) {
            tempTurns.get(i).setIndex(i); // reset the index of the turn
        }
        this.turns = tempTurns;
    }

    @Override
    public String toString() {
        return "conversation: " + this.id
                + ". # speakers: " + this.getNumSpeakers()
                + ". # turns: " + this.getNumTurns();
    }
}
